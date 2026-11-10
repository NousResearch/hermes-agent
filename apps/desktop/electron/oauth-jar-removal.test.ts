import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { expect, it, onTestFinished } from 'vitest'

import { type ConnectionRegistry, normalizeRegistry, removeConnection } from './connection-registry'
import { createGatewayConnectionsStore } from './gateway-connections-store'
import { connectionConfigUrls, type CookieSessionLike, createRemovedConnectionCookieCleaner } from './oauth-jar-removal'
import { LEGACY_OAUTH_PARTITION, resolveOauthPartition } from './oauth-partition'

// #98242: a removed OAuth gateway's sign-in must not outlive it in any jar it rode.
interface Cookie {
  domain: string
  name: string
  path: string
  secure: boolean
}

/** Partition -> cookies, behind the slice of Electron's Session API removal uses. */
function cookieJars(sharedUrls: () => unknown[] = () => ['https://portal.nousresearch.com']) {
  const jars = new Map<string, Cookie[]>()
  const mirrorCleared: string[] = []

  const sessionFor = (partition: string): CookieSessionLike => ({
    cookies: {
      get: async () => [...(jars.get(partition) || [])],
      remove: async (url, name) => {
        const { hostname, pathname } = new URL(url)
        jars.set(
          partition,
          (jars.get(partition) || []).filter(
            c => !(c.name === name && c.domain.replace(/^\./, '') === hostname && c.path === pathname)
          )
        )
      }
    }
  })

  const seed = (partition: string, domain: string, name: string) =>
    jars.set(partition, [...(jars.get(partition) || []), { domain, name, path: '/', secure: true }])

  const names = (partition: string) => (jars.get(partition) || []).map(c => `${c.domain}:${c.name}`)

  const clean = createRemovedConnectionCookieCleaner({
    sessionFor,
    clearMirror: (partition, url) => mirrorCleared.push(`${partition}|${url ?? '*'}`),
    v1RemoteUrl: () => undefined,
    sharedUrls
  })

  return { clean, mirrorCleared, names, seed }
}

const URL_A = 'https://gw.example.invalid'
const gateway = { id: 'remote-a', kind: 'remote', label: 'A', url: URL_A, authMode: 'oauth' }
const other = { id: 'remote-b', kind: 'remote', label: 'B', url: 'https://other.example.invalid', authMode: 'oauth' }

const list = (primary: string, connections: object[]) =>
  normalizeRegistry({
    primary,
    lastUsed: primary,
    connections: [{ id: 'local', kind: 'local', label: 'This device' }, ...connections]
  })

it('removing a gateway that went secondary -> primary clears both jars, so a re-add with its id is signed out', async () => {
  const jars = cookieJars()
  const secondary = list('local', [gateway, other])
  const primary = list('remote-a', [gateway, other])
  const privateJar = resolveOauthPartition(URL_A, { registry: secondary })
  expect(privateJar).not.toBe(LEGACY_OAUTH_PARTITION)
  expect(resolveOauthPartition(URL_A, { registry: primary })).toBe(LEGACY_OAUTH_PARTITION)

  jars.seed(privateJar, 'gw.example.invalid', 'hermes_session')
  jars.seed(LEGACY_OAUTH_PARTITION, 'gw.example.invalid', 'hermes_session')
  jars.seed(LEGACY_OAUTH_PARTITION, 'portal.nousresearch.com', 'portal_session')
  const otherJar = resolveOauthPartition(other.url, { registry: primary })
  jars.seed(otherJar, 'other.example.invalid', 'hermes_session')

  await jars.clean('remote-a', primary)

  // Re-added as a secondary (recycled id) or as primary: no sign-in either way.
  expect(jars.names(resolveOauthPartition(URL_A, { registry: secondary }))).toEqual([])
  expect(jars.names(LEGACY_OAUTH_PARTITION)).toEqual(['portal.nousresearch.com:portal_session'])
  expect(jars.names(otherJar)).toEqual(['other.example.invalid:hermes_session'])
  expect(jars.mirrorCleared).toEqual([`${privateJar}|*`, `${LEGACY_OAUTH_PARTITION}|${URL_A}`])
})

it('a removal another writer (the Webapp) made reaches the cleanup with the list that still held it', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-oauth-jar-removal-'))
  onTestFinished(() => fs.rmSync(root, { force: true, recursive: true }))
  const userData = path.join(root, 'userData')
  fs.mkdirSync(userData)

  const store = (registry: ConnectionRegistry) =>
    createGatewayConnectionsStore({ root, userData, legacyRegistry: () => registry, decrypt: () => '' })

  const desktop = store(list('remote-a', [gateway, other]))
  desktop.read()

  const jars = cookieJars()
  jars.seed(
    resolveOauthPartition(URL_A, { registry: list('local', [gateway]) }),
    'gw.example.invalid',
    'hermes_session'
  )
  jars.seed(LEGACY_OAUTH_PARTITION, 'gw.example.invalid', 'hermes_session')
  let cleaned: Promise<void> = Promise.resolve()

  // main.ts's watcher handler, removal branch.
  const stop = desktop.watch({
    onRegistry: (changes, { after, before }) => {
      for (const change of changes.filter(c => c.reason === 'removed')) {
        cleaned = jars.clean(change.connectionId, before, after)
      }
    },
    onTokens: () => undefined
  })

  onTestFinished(stop)

  // fs.watchFile takes its baseline stat asynchronously, so a write that lands before it becomes the
  // baseline instead of a change. Re-issue the (idempotent) outside removal until the watcher reacts;
  // main.ts registers the watcher at load, long before any outside edit.
  const webapp = store(list('local', []))
  await expect
    .poll(
      () => {
        if (!jars.mirrorCleared.length) {
          webapp.write(removeConnection(webapp.read(), 'remote-a'))
        }

        return jars.mirrorCleared.length
      },
      { interval: 250, timeout: 12_000 }
    )
    .toBe(2)
  await cleaned
  expect(jars.names(resolveOauthPartition(URL_A, { registry: list('local', [gateway]) }))).toEqual([])
  expect(jars.names(LEGACY_OAUTH_PARTITION)).toEqual([])
}, 20_000)

it("a connection.json per-profile route on the removed gateway's host keeps its shared-jar sign-in", async () => {
  const config = { remote: {}, profiles: { coder: { mode: 'remote', url: 'https://gw.example.invalid:9000' } } }
  const jars = cookieJars(() => ['https://portal.nousresearch.com', ...connectionConfigUrls(config)])
  const secondary = { ...gateway, url: 'https://gw.example.invalid:8000' }
  jars.seed(LEGACY_OAUTH_PARTITION, 'gw.example.invalid', 'hermes_session')

  await jars.clean('remote-a', list('local', [secondary]))

  expect(jars.names(LEGACY_OAUTH_PARTITION)).toEqual(['gw.example.invalid:hermes_session'])
})

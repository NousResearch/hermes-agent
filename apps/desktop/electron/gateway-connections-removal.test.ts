import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'

import { afterEach, test } from 'vitest'

import { normalizeRegistry, removeConnection } from './connection-registry'
import { forgetRemovedGlobalRoute, globalRouteRemoved } from './connection-registry-remove'
import { createDesktopGatewayConnectionsStore } from './gateway-connections-store'

// #142906 / #150084: a saved gateway the user removed came back as primary on the
// next fresh read, because connection.json still named it. The old userData files
// feed a ONE-TIME import; after that the shared list is the authority, for Desktop
// and for the Webapp alike.
const repo = path.resolve(import.meta.dirname, '../../..')
const python = process.env.HERMES_PYTHON || 'python3'
const dirs: string[] = []

afterEach(() => {
  for (const dir of dirs.splice(0)) {
    fs.rmSync(dir, { force: true, recursive: true })
  }
})

const STALE_V1 = { mode: 'remote', remote: { url: 'http://127.0.0.1:9119', authMode: 'token', token: 'x' } }

function machine() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-gw-removal-'))
  dirs.push(dir)
  const root = path.join(dir, 'hermes')
  const userData = path.join(dir, 'userData')
  fs.mkdirSync(root, { recursive: true })
  fs.mkdirSync(userData, { recursive: true })
  const v1File = path.join(userData, 'connection.json')
  fs.writeFileSync(v1File, JSON.stringify(STALE_V1))

  fs.writeFileSync(
    path.join(userData, 'connections.json'),
    JSON.stringify({
      version: 2,
      primary: 'old-http',
      launchMode: 'primary',
      lastUsed: 'old-http',
      connections: [
        { id: 'local', kind: 'local', label: 'This device' },
        { id: 'old-http', kind: 'remote', label: 'Old HTTP', url: 'http://127.0.0.1:9119', authMode: 'token' },
        { id: 'box', kind: 'ssh', label: 'Box', host: 'box.lan', user: 'me' }
      ]
    })
  )

  const readV1 = () => JSON.parse(fs.readFileSync(v1File, 'utf8'))

  // A fresh store per call is a fresh Desktop launch: no cache survives.
  const launch = () =>
    createDesktopGatewayConnectionsStore({ root, userData, readV1Config: readV1, decrypt: (s: any) => s?.value || '' })

  return { dir, launch, readV1, root, v1File }
}

test('a removed gateway stays removed across relaunches while the old connection.json still names it', () => {
  const m = machine()
  const first = m.launch()
  const imported = first.read()
  assert.deepEqual(
    imported.connections.map(c => c.id),
    ['local', 'old-http', 'box']
  )

  first.write({ ...removeConnection(imported, 'old-http'), primary: 'box' })

  for (let relaunch = 0; relaunch < 2; relaunch++) {
    const registry = m.launch().read()
    assert.deepEqual(
      registry.connections.map(c => c.id),
      ['local', 'box'],
      'the removed gateway was re-added'
    )
    assert.equal(registry.primary, 'box', 'the removed gateway was made primary again')
  }

  // The old file is never touched by the store itself.
  assert.deepEqual(m.readV1(), STALE_V1)
})

test('a gateway the Webapp removed stays removed for Desktop, and its stale route is reset', async () => {
  const m = machine()
  m.launch().read()

  const env = { HERMES_HOME: m.root, HOME: m.dir, PATH: process.env.PATH, PYTHONPATH: repo }
  const script = 'from hermes_cli import gateway_connections as g\ng.remove_connection("old-http")'
  await promisify(execFile)(python, ['-c', script], { env })

  const registry = m.launch().read()
  assert.deepEqual(
    registry.connections.map(c => c.id),
    ['local', 'box']
  )

  const writes: unknown[] = []

  const io = {
    readConfig: m.readV1,
    writeConfig: (config: Record<string, any>) => {
      writes.push(config)
      fs.writeFileSync(m.v1File, JSON.stringify(config))
    }
  }

  assert.equal(forgetRemovedGlobalRoute(registry, io), true)
  assert.equal(m.readV1().mode, 'local', 'Desktop would keep dialing the removed gateway at boot')
  // Idempotent, and a live route the list still holds is never touched.
  assert.equal(forgetRemovedGlobalRoute(registry, io), false)
  fs.writeFileSync(m.v1File, JSON.stringify({ mode: 'ssh', remote: { host: 'box.lan', user: 'me' } }))
  assert.equal(forgetRemovedGlobalRoute(registry, io), false)
  assert.equal(writes.length, 1)
})

test('an SSH route is the same gateway only with the same remote profile', () => {
  const registry = normalizeRegistry({
    connections: [
      { id: 'local', kind: 'local', label: 'This device' },
      { id: 'box-coder', kind: 'ssh', label: 'Box coder', host: 'box.lan', user: 'me', remoteProfile: 'coder' }
    ]
  })

  const route = (remoteProfile?: string) => ({
    mode: 'ssh',
    remote: { host: 'box.lan', user: 'me', port: 22, remoteProfile }
  })
  assert.equal(globalRouteRemoved(registry, route('coder')), false)
  assert.equal(globalRouteRemoved(registry, route('writer')), true, 'the removed profile route survived on a sibling')
  assert.equal(globalRouteRemoved(registry, { mode: 'local', remote: {} }), false)
})

test('an unreadable store never resets connection.json', () => {
  // Mode 000 does not stop the owner reading on Windows, nor root anywhere.
  if (process.platform === 'win32' || process.getuid?.() === 0) {
    return
  }

  const m = machine()
  m.launch().read()
  const store = path.join(m.root, 'gateway-connections.json')
  fs.chmodSync(store, 0)
  const launch = m.launch()

  try {
    assert.throws(() => launch.readStrict())
    assert.deepEqual(
      launch.read().connections.map(c => c.id),
      ['local'],
      'read() still answers the local-only fallback'
    )
  } finally {
    fs.chmodSync(store, 0o600)
  }
})

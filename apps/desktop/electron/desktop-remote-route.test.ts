import assert from 'node:assert/strict'

import { test } from 'vitest'

import { normalizeRegistry, REGISTRY_VERSION } from './connection-registry'
import { resolveDesktopRemoteRoute } from './desktop-remote-route'

const tokenA = { encoding: 'plain', value: 'token-a' }
const tokenB = { encoding: 'plain', value: 'token-b' }

function registry(primary: string, connections: Record<string, unknown>[]) {
  return normalizeRegistry({
    version: REGISTRY_VERSION,
    primary,
    connections: [{ id: 'local', kind: 'local', label: 'This device' }, ...connections]
  })
}

test('profile remote wins precedence and carries one exact registry id', () => {
  const route = resolveDesktopRemoteRoute({
    config: {
      mode: 'remote',
      remote: { url: 'https://global.test', authMode: 'token', token: tokenB },
      profiles: {
        worker: { mode: 'remote', url: 'https://worker.test/', authMode: 'token', token: tokenA }
      }
    },
    env: { url: 'https://env.test', token: 'env-token' },
    profile: 'worker',
    registry: registry('global', [
      { id: 'global', kind: 'remote', label: 'Global', url: 'https://global.test', token: tokenB },
      { id: 'worker', kind: 'remote', label: 'Worker', url: 'https://worker.test', token: tokenA }
    ])
  })

  assert.equal(route?.kind, 'remote')
  assert.equal(route?.source, 'profile')
  assert.equal(route?.connectionId, 'worker')
})

test('environment route wins over global but never claims a registry id', () => {
  const route = resolveDesktopRemoteRoute({
    config: { mode: 'remote', remote: { url: 'https://global.test', token: tokenB } },
    env: { url: 'https://env.test', token: 'token-a' },
    registry: registry('env', [{ id: 'env', kind: 'remote', label: 'Env', url: 'https://env.test', token: tokenA }])
  })

  assert.equal(route?.source, 'env')
  assert.equal(route?.connectionId, undefined)
  assert.equal(route?.kind === 'remote' ? route.url : null, 'https://env.test')
})

test('environment URL without its token keeps the existing error', () => {
  assert.throws(
    () =>
      resolveDesktopRemoteRoute({
        config: { mode: 'local' },
        env: { url: 'https://env.test' },
        registry: registry('local', [])
      }),
    /HERMES_DESKTOP_REMOTE_TOKEN is not/
  )
})

test('global remote uses exact primary provenance when another row is identical', () => {
  const route = resolveDesktopRemoteRoute({
    config: { mode: 'remote', remote: { url: 'https://gateway.test/', authMode: 'token', token: tokenA } },
    registry: registry('gateway-primary', [
      { id: 'gateway-primary', kind: 'remote', label: 'Primary', url: 'https://gateway.test', token: tokenA },
      { id: 'gateway-copy', kind: 'remote', label: 'Copy', url: 'https://gateway.test', token: tokenA }
    ])
  })

  assert.equal(route?.source, 'settings')
  assert.equal(route?.connectionId, 'gateway-primary')
})

test('global route fails closed when primary differs, even if another row matches', () => {
  const route = resolveDesktopRemoteRoute({
    config: { mode: 'remote', remote: { url: 'https://gateway.test', authMode: 'token', token: tokenA } },
    registry: registry('other', [
      { id: 'other', kind: 'remote', label: 'Other', url: 'https://other.test', token: tokenB },
      { id: 'matching', kind: 'remote', label: 'Matching', url: 'https://gateway.test', token: tokenA }
    ])
  })

  assert.equal(route?.connectionId, undefined)
})

test('profile SSH identity includes port, key, paths, and remote profile', () => {
  const ssh = {
    mode: 'ssh',
    host: 'box.test',
    user: 'hermes',
    port: 2222,
    keyPath: '/keys/a',
    remoteHermesPath: '/srv/hermes',
    remoteProfile: 'worker'
  }

  const route = resolveDesktopRemoteRoute({
    config: { mode: 'local', profiles: { worker: ssh } },
    profile: 'worker',
    registry: registry('local', [
      { id: 'wrong-port', kind: 'ssh', label: 'Wrong port', ...ssh, port: 22 },
      { id: 'worker-ssh', kind: 'ssh', label: 'Worker SSH', ...ssh }
    ])
  })

  assert.equal(route?.kind, 'ssh')
  assert.equal(route?.connectionId, 'worker-ssh')
})

test('profile SSH route fails closed when any dial field differs', () => {
  const ssh = {
    mode: 'ssh',
    host: 'box.test',
    user: 'hermes',
    port: 2222,
    keyPath: '/keys/a',
    remoteHermesPath: '/srv/hermes',
    remoteProfile: 'worker'
  }

  const variants = [
    { ...ssh, port: 2200 },
    { ...ssh, keyPath: '/keys/b' },
    { ...ssh, remoteHermesPath: '/opt/hermes' },
    { ...ssh, remoteProfile: 'default' },
    { ...ssh, user: 'other' }
  ]

  for (const [index, variant] of variants.entries()) {
    const route = resolveDesktopRemoteRoute({
      config: { mode: 'local', profiles: { worker: ssh } },
      profile: 'worker',
      registry: registry('local', [{ id: `ssh-${index}`, kind: 'ssh', label: `SSH ${index}`, ...variant }])
    })

    assert.equal(route?.connectionId, undefined)
  }
})

test('global SSH treats an omitted port as 22 and checks the primary route', () => {
  const route = resolveDesktopRemoteRoute({
    config: { mode: 'ssh', remote: { mode: 'ssh', host: 'box.test', user: 'hermes' } },
    registry: registry('ssh-primary', [
      { id: 'ssh-primary', kind: 'ssh', label: 'SSH primary', host: 'box.test', user: 'hermes', port: 22 }
    ])
  })

  assert.equal(route?.kind, 'ssh')
  assert.equal(route?.connectionId, 'ssh-primary')
})

test('profile route omits identity when two registry entries match exactly', () => {
  const block = { mode: 'remote', url: 'https://worker.test', authMode: 'token', token: tokenA }

  const route = resolveDesktopRemoteRoute({
    config: { mode: 'local', profiles: { worker: block } },
    profile: 'worker',
    registry: registry('local', [
      { id: 'worker-a', kind: 'remote', label: 'Worker A', ...block },
      { id: 'worker-b', kind: 'remote', label: 'Worker B', ...block }
    ])
  })

  assert.equal(route?.connectionId, undefined)
})

test('kind, auth material, headers, and Cloud org stay part of route identity', () => {
  const cloud = {
    mode: 'cloud',
    url: 'https://cloud.test',
    authMode: 'oauth',
    headers: { 'CF-Access': { encoding: 'plain', value: 'a' } },
    org: 'org-a'
  }

  const route = resolveDesktopRemoteRoute({
    config: { mode: 'cloud', remote: cloud },
    registry: registry('cloud', [
      { id: 'cloud', kind: 'cloud', label: 'Cloud', ...cloud },
      { id: 'remote', kind: 'remote', label: 'Remote', ...cloud },
      { id: 'other-org', kind: 'cloud', label: 'Other org', ...cloud, org: 'org-b' }
    ])
  })

  assert.equal(route?.kind, 'cloud')
  assert.equal(route?.connectionId, 'cloud')
})

test('URL route fails closed for different token, headers, kind, or Cloud org', () => {
  const cases = [
    {
      config: { mode: 'remote', remote: { url: 'https://gateway.test', token: tokenA } },
      primary: { kind: 'remote', url: 'https://gateway.test', token: tokenB }
    },
    {
      config: {
        mode: 'remote',
        remote: {
          url: 'https://gateway.test',
          token: tokenA,
          headers: { 'CF-Access': { encoding: 'plain', value: 'a' } }
        }
      },
      primary: {
        kind: 'remote',
        url: 'https://gateway.test',
        token: tokenA,
        headers: { 'CF-Access': { encoding: 'plain', value: 'b' } }
      }
    },
    {
      config: { mode: 'remote', remote: { url: 'https://gateway.test', token: tokenA } },
      primary: { kind: 'cloud', url: 'https://gateway.test', token: tokenA }
    },
    {
      config: { mode: 'cloud', remote: { url: 'https://gateway.test', authMode: 'oauth', org: 'a' } },
      primary: { kind: 'cloud', url: 'https://gateway.test', authMode: 'oauth', org: 'b' }
    }
  ]

  for (const [index, item] of cases.entries()) {
    const route = resolveDesktopRemoteRoute({
      config: item.config,
      registry: registry('primary', [{ id: 'primary', label: `Primary ${index}`, ...item.primary }])
    })

    assert.equal(route?.connectionId, undefined)
  }
})

test('local config without overrides returns null', () => {
  assert.equal(resolveDesktopRemoteRoute({ config: { mode: 'local' }, registry: registry('local', []) }), null)
})

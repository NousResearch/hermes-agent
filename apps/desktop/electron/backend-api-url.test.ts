import assert from 'node:assert/strict'

import { test } from 'vitest'

import { joinBackendApiUrl } from './backend-api-url'

const LOCAL = 'http://127.0.0.1:7890'
const REMOTE = 'https://gateway.example.com'

test('joinBackendApiUrl keeps local API paths on the loopback origin', () => {
  assert.equal(joinBackendApiUrl(LOCAL, '/api/status'), `${LOCAL}/api/status`)
  assert.equal(joinBackendApiUrl(LOCAL, '/api/sessions?limit=50&offset=0'), `${LOCAL}/api/sessions?limit=50&offset=0`)
})

test('joinBackendApiUrl keeps remote gateway paths on the configured origin', () => {
  assert.equal(joinBackendApiUrl(REMOTE, '/api/sessions'), `${REMOTE}/api/sessions`)
  assert.equal(joinBackendApiUrl(`${REMOTE}/`, '/api/model/info?profile=iris'), `${REMOTE}/api/model/info?profile=iris`)
})

test('joinBackendApiUrl allows path segments that look like userinfo but stay on-origin', () => {
  assert.equal(joinBackendApiUrl(LOCAL, '/@literal'), `${LOCAL}/@literal`)
})

test('joinBackendApiUrl rejects @-authority retargeting that would leak credentials', () => {
  assert.throws(() => joinBackendApiUrl(LOCAL, '@attacker.example/'), /relative path/)
  assert.throws(() => joinBackendApiUrl(REMOTE, '@attacker.example/steal'), /relative path/)
})

test('joinBackendApiUrl rejects whitespace-prefixed @-authority retargeting', () => {
  assert.throws(() => joinBackendApiUrl(LOCAL, ' @attacker.example/'), /relative path/)
  assert.throws(() => joinBackendApiUrl(LOCAL, '\t@attacker.example/'), /relative path/)
})

test('joinBackendApiUrl rejects protocol-relative-shaped and empty paths', () => {
  // Shape reject: WHATWG concat after host:port keeps //evil on-origin, but
  // renderer paths must still be single-slash relative API paths.
  assert.throws(() => joinBackendApiUrl(LOCAL, '//attacker.example/'), /relative path/)
  assert.throws(() => joinBackendApiUrl(LOCAL, ''), /non-empty/)
  assert.throws(() => joinBackendApiUrl(LOCAL, null), /non-empty/)
  assert.throws(() => joinBackendApiUrl(LOCAL, undefined), /non-empty/)
})

test('joinBackendApiUrl rejects absolute URLs smuggled as the path', () => {
  assert.throws(() => joinBackendApiUrl(LOCAL, 'http://attacker.example/'), /relative path/)
  assert.throws(() => joinBackendApiUrl(LOCAL, 'https://attacker.example/'), /relative path/)
})

// The hermes:api handler routes the renderer path through these rewrites before
// fetchJsonForBackend joins it; whatever they emit must still be refused or on-origin.
test('routed renderer paths never leave the backend origin once joined', async () => {
  const { pathForRegistryBackendRequest, resolveProfileApiRequest } = await import('./connection-config')
  const hostile = [
    '@attacker.example/',
    ' @attacker.example/',
    'attacker.example',
    '//attacker.example/',
    'https://attacker.example/'
  ]

  for (const path of hostile) {
    const routed = [
      resolveProfileApiRequest('iris', path, { globalRemote: true, primaryProfile: 'default' }).requestPath,
      resolveProfileApiRequest('iris', path, {}).requestPath,
      pathForRegistryBackendRequest(path, 'iris', { sharedRemote: true }),
      pathForRegistryBackendRequest(path, 'iris', { mode: 'ssh', remoteProfile: 'default' })
    ]

    for (const requestPath of routed) {
      let joined: null | string = null

      try {
        joined = joinBackendApiUrl(REMOTE, requestPath)
      } catch {
        continue
      }

      assert.equal(new URL(joined).origin, REMOTE, `${JSON.stringify(path)} -> ${requestPath}`)
    }
  }
})

test('joinBackendApiUrl keeps gateway file download URLs on the backend origin', async () => {
  const { gatewayFileRequestPaths } = await import('./gateway-file-download')

  for (const filePath of ['/home/u/r&d #1.pdf', '@evil.example/x', '//evil.example/x', 'http://evil.example/']) {
    for (const sessionId of [undefined, 's1', '@evil.example/', '#@evil.example']) {
      const { download } = gatewayFileRequestPaths(filePath, path => path, sessionId)
      const joined = new URL(joinBackendApiUrl(`${REMOTE}/`, download))

      assert.equal(joined.origin, REMOTE)
      assert.equal(joined.pathname, '/api/fs/download')
      assert.equal(joined.searchParams.get('path'), filePath)
      assert.equal(joined.searchParams.get('session_id'), sessionId ?? null)
    }
  }
})

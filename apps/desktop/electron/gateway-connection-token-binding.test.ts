import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, test } from 'vitest'

import { removeConnection, upsertConnection } from './connection-registry'
import { createGatewayConnectionsStore, GATEWAY_CONNECTION_TOKENS_FILE } from './gateway-connections-store'
import { loadNativeTokenSet, persistNativeTokenSet } from './native-token-store'

// A native sign-in belongs to the gateway that issued it. Connection ids are recycled label
// slugs, so neither a URL edit nor remove + re-add may hand one gateway's tokens to another.
const dirs: string[] = []

afterEach(() => {
  for (const dir of dirs.splice(0)) {
    fs.rmSync(dir, { force: true, recursive: true })
  }
})

function storeIn() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-gw-bind-'))
  dirs.push(dir)
  const root = path.join(dir, 'hermes')
  const userData = path.join(dir, 'userData')
  fs.mkdirSync(root, { recursive: true })
  fs.mkdirSync(userData, { recursive: true })

  const store = createGatewayConnectionsStore({
    root,
    userData,
    legacyRegistry: () => ({ version: 2, primary: 'local', launchMode: 'primary', lastUsed: 'local',
      connections: [{ id: 'local', kind: 'local', label: 'This device' }] }) as any,
    decrypt: (secret: any) => String(secret?.value || '')
  })

  return { root, store }
}

const set = (access: string) => ({ accessToken: access, refreshToken: `rt-${access}`, expiresAt: 99, provider: 'p', userId: 'u' })
const mac = (url: string) => ({ id: 'mac-mini', kind: 'remote', label: 'Mac mini', url, authMode: 'oauth' }) as any

test('a re-pointed or removed-and-re-added connection never loads the old gateway’s sign-in', () => {
  const { root, store } = storeIn()
  store.write(upsertConnection(store.read(), mac('https://a.example')))
  persistNativeTokenSet('https://a.example', set('AT-A'), store.nativeTokenIo())
  assert.deepEqual(loadNativeTokenSet('https://a.example', store.nativeTokenIo()), set('AT-A'))

  store.write(upsertConnection(store.read(), mac('https://b.example')))
  assert.equal(loadNativeTokenSet('https://b.example', store.nativeTokenIo()), null)

  persistNativeTokenSet('https://b.example', set('AT-B'), store.nativeTokenIo())
  store.write(removeConnection(store.read(), 'mac-mini'))
  store.write(upsertConnection(store.read(), mac('https://c.example')))
  assert.equal(loadNativeTokenSet('https://c.example', store.nativeTokenIo()), null)
  assert.deepEqual(JSON.parse(fs.readFileSync(path.join(root, GATEWAY_CONNECTION_TOKENS_FILE), 'utf8')).tokens, {})
})

test('a set stored under an id for another URL (an older writer) is never handed out', () => {
  const { root, store } = storeIn()
  store.write(upsertConnection(store.read(), mac('https://b.example')))
  fs.writeFileSync(
    path.join(root, GATEWAY_CONNECTION_TOKENS_FILE),
    JSON.stringify({ version: 1, revision: 1, tokens: { 'mac-mini': { ...set('AT-A'), baseUrl: 'https://a.example' } } })
  )
  assert.equal(loadNativeTokenSet('https://b.example', store.nativeTokenIo()), null)
})

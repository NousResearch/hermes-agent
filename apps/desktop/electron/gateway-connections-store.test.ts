import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'

import { afterEach, test } from 'vitest'

import { mergeConnectionInput, normalizeConnectionInput, normalizeRegistry } from './connection-registry'
import {
  createGatewayConnectionsStore,
  GATEWAY_CONNECTION_TOKENS_FILE,
  GATEWAY_CONNECTIONS_FILE,
  readLegacyUserDataRegistry,
  registryChanges
} from './gateway-connections-store'
import { loadNativeTokenSet, persistNativeTokenSet } from './native-token-store'

// Desktop (this module) and the Webapp's Python server (hermes_cli/gateway_connections.py)
// keep ONE saved-gateways list. Existing Desktop users must keep everything on the move.
const repo = path.resolve(import.meta.dirname, '../../..')
const python = process.env.HERMES_PYTHON || 'python3'

const fixture = JSON.parse(
  fs.readFileSync(path.join(repo, 'tests/fixtures/gateway_connections/normalize_cases.json'), 'utf8')
)

const dirs: string[] = []

afterEach(() => {
  for (const dir of dirs.splice(0)) {
    fs.rmSync(dir, { force: true, recursive: true })
  }
})

function scratch() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-gw-conns-'))
  dirs.push(dir)
  const root = path.join(dir, 'hermes')
  const userData = path.join(dir, 'userData')
  fs.mkdirSync(root, { recursive: true })
  fs.mkdirSync(userData, { recursive: true })

  return { dir, root, userData }
}

// safeStorage stand-in: "enc:<x>" opens to <x>; "locked:" never opens.
const decrypt = (secret: any) => {
  if (!secret || typeof secret !== 'object') {
    return ''
  }

  if (secret.encoding !== 'safeStorage') {
    return String(secret.value || '')
  }

  const value = Buffer.from(String(secret.value || ''), 'base64').toString('utf8')

  return value.startsWith('enc:') ? value.slice(4) : ''
}

const sealed = (plain: string) => ({ encoding: 'safeStorage', value: Buffer.from(`enc:${plain}`).toString('base64') })
const locked = { encoding: 'safeStorage', value: Buffer.from('locked:x').toString('base64') }

function storeFor(root: string, userData: string, v1: unknown = {}) {
  return createGatewayConnectionsStore({
    root,
    userData,
    legacyRegistry: () => readLegacyUserDataRegistry(userData, v1),
    decrypt
  })
}

const readJson = (file: string) => JSON.parse(fs.readFileSync(file, 'utf8'))
const writeJson = (file: string, doc: unknown) => fs.writeFileSync(file, JSON.stringify(doc))

const tokenSet = (access: string) => ({
  accessToken: access,
  refreshToken: `rt-${access}`,
  expiresAt: 99,
  provider: 'p',
  userId: 'u'
})

test('the TypeScript normalizer produces the shared fixture the Python twin is held to', () => {
  for (const [name, raw] of Object.entries(fixture.inputs)) {
    assert.deepEqual(normalizeRegistry(raw), fixture.expected[name], name)
  }

  const registry = normalizeRegistry(fixture.saveRegistry)

  fixture.saves.forEach((input: any, index: number) => {
    const existing = input.id ? registry.connections.find(c => c.id === input.id) : null
    let got

    try {
      got = { entry: normalizeConnectionInput(mergeConnectionInput(input, existing), registry) }
    } catch (error) {
      got = { error: (error as Error).message }
    }

    assert.deepEqual(got, fixture.expectedSaves[index])
  })
})

test('a v1-only user (connection.json, no connections.json) migrates with the token made plain', () => {
  const { root, userData } = scratch()

  const v1 = {
    mode: 'remote',
    remote: {
      url: 'https://gw.example:8443',
      authMode: 'token',
      token: sealed('v1-token'),
      headers: { 'X-A': sealed('ha') }
    },
    profiles: { coder: { mode: 'ssh', host: 'alice@box.lan:2222', remoteProfile: 'coder' } }
  }

  writeJson(path.join(userData, 'connection.json'), v1)
  const registry = storeFor(root, userData, v1).read()
  const doc = readJson(path.join(root, GATEWAY_CONNECTIONS_FILE))

  assert.equal(doc.primary, 'gw-example-8443')
  assert.deepEqual(doc.connections[1], {
    id: 'gw-example-8443',
    kind: 'remote',
    label: 'gw.example:8443',
    url: 'https://gw.example:8443',
    authMode: 'token',
    token: { encoding: 'plain', value: 'v1-token' },
    headers: { 'X-A': { encoding: 'plain', value: 'ha' } }
  })
  assert.deepEqual(doc.connections[2], {
    id: 'box-lan',
    kind: 'ssh',
    label: 'box.lan',
    host: 'box.lan',
    user: 'alice',
    port: 2222,
    remoteProfile: 'coder'
  })
  assert.equal(registry.connections.length, 3)
  assert.equal(fs.statSync(path.join(root, GATEWAY_CONNECTIONS_FILE)).mode & 0o777, 0o600)
  assert.equal(fs.existsSync(path.join(userData, 'connections.json')), false, 'userData is never written')
})

test('a v2 user keeps every connection, field, secret, quarantined entry and sign-in', () => {
  const { root, userData } = scratch()

  const legacy = {
    version: 2,
    primary: 'homelab',
    launchMode: 'last-used',
    lastUsed: 'box',
    connections: [
      { id: 'local', kind: 'local', label: 'Studio Mac' },
      {
        id: 'homelab',
        kind: 'remote',
        label: 'Homelab',
        url: 'http://10.0.0.5:9119',
        authMode: 'token',
        token: sealed('tok-1'),
        headers: { 'CF-Access-Client-Id': sealed('cid'), 'CF-Access-Client-Secret': locked }
      },
      { id: 'gw', kind: 'remote', label: 'OAuth gw', url: 'https://gw.example', authMode: 'oauth' },
      {
        id: 'cloud',
        kind: 'cloud',
        label: 'Cloud',
        url: 'https://c.example',
        authMode: 'oauth',
        org: 'acme',
        name: 'inst'
      },
      { id: 'stuck', kind: 'remote', label: 'Stuck', url: 'http://stuck:1', authMode: 'token', token: locked },
      {
        id: 'box',
        kind: 'ssh',
        label: 'Box',
        host: 'box.lan',
        user: 'alice',
        port: 2222,
        keyPath: '~/.ssh/id',
        remoteHermesPath: '/opt/h',
        remoteProfile: 'coder',
        token: sealed('ssh-tok')
      }
    ],
    quarantined: [{ reason: 'entry-missing-url', entry: { kind: 'remote', label: 'Broken', token: sealed('q-tok') } }]
  }

  writeJson(path.join(userData, 'connections.json'), legacy)
  writeJson(path.join(userData, 'native-oauth-tokens.json'), {
    'https://gw.example': sealed(JSON.stringify(tokenSet('gw-at'))),
    'https://c.example': { encoding: 'plain', value: JSON.stringify(tokenSet('cloud-at')) },
    'https://pending.example': sealed(JSON.stringify(tokenSet('pending-at'))),
    'https://dead.example': locked
  })
  const before = fs.readFileSync(path.join(userData, 'connections.json'), 'utf8')

  const store = storeFor(root, userData)
  store.read()
  const doc = readJson(path.join(root, GATEWAY_CONNECTIONS_FILE))

  const plainLegacy = JSON.parse(JSON.stringify(legacy), (key, value) =>
    value && typeof value === 'object' && value.encoding === 'safeStorage' && decrypt(value)
      ? { encoding: 'plain', value: decrypt(value) }
      : value
  )

  for (const key of ['primary', 'launchMode', 'lastUsed', 'connections', 'quarantined']) {
    assert.deepEqual(doc[key], plainLegacy[key], key)
  }

  assert.deepEqual(doc.importNotices, [
    { connectionId: 'homelab', field: 'headers.CF-Access-Client-Secret', reason: 'undecryptable' },
    { connectionId: 'stuck', field: 'token', reason: 'undecryptable' }
  ])

  const tokens = readJson(path.join(root, GATEWAY_CONNECTION_TOKENS_FILE))
  // Each imported set keeps the URL main's file keyed it by: it is bound to that gateway.
  assert.deepEqual(tokens.tokens, {
    gw: { ...tokenSet('gw-at'), baseUrl: 'https://gw.example' },
    cloud: { ...tokenSet('cloud-at'), baseUrl: 'https://c.example' }
  })
  assert.deepEqual(tokens.unbound, { 'https://pending.example': tokenSet('pending-at') })
  assert.deepEqual(tokens.undecrypted, { 'https://dead.example': locked })

  // Signed in with no re-login: main's loader finds each set through the new io.
  const io = store.nativeTokenIo()
  assert.deepEqual(loadNativeTokenSet('https://gw.example', io), tokenSet('gw-at'))
  assert.deepEqual(loadNativeTokenSet('https://pending.example', io), tokenSet('pending-at'))
  persistNativeTokenSet('https://gw.example', tokenSet('gw-2'), io)
  assert.deepEqual(readJson(path.join(root, GATEWAY_CONNECTION_TOKENS_FILE)).tokens.gw, {
    ...tokenSet('gw-2'),
    baseUrl: 'https://gw.example'
  })

  assert.equal(fs.readFileSync(path.join(userData, 'connections.json'), 'utf8'), before, 'userData untouched')
})

test('a store the Webapp created first is merged into, never overwritten, and never re-imported', () => {
  const { root, userData } = scratch()
  writeJson(path.join(root, GATEWAY_CONNECTIONS_FILE), {
    version: 2,
    revision: 3,
    primary: 'local',
    launchMode: 'primary',
    lastUsed: 'local',
    connections: [
      { id: 'local', kind: 'local', label: 'This device' },
      {
        id: 'web',
        kind: 'remote',
        label: 'Web added',
        url: 'http://web:1',
        authMode: 'token',
        token: { encoding: 'plain', value: 'w' }
      },
      { id: 'homelab', kind: 'remote', label: 'Same URL', url: 'http://10.0.0.5:9119/', authMode: 'token' }
    ]
  })
  writeJson(path.join(userData, 'connections.json'), {
    version: 2,
    primary: 'homelab',
    launchMode: 'last-used',
    lastUsed: 'web',
    connections: [
      { id: 'local', kind: 'local', label: 'Studio' },
      {
        id: 'homelab',
        kind: 'remote',
        label: 'Homelab',
        url: 'http://10.0.0.5:9119',
        authMode: 'token',
        token: sealed('t')
      },
      { id: 'web', kind: 'remote', label: 'Web added', url: 'http://other:1', authMode: 'token', token: sealed('o') },
      { id: 'box', kind: 'ssh', label: 'Box', host: 'box.lan' }
    ]
  })

  const store = storeFor(root, userData)
  const merged = store.read()

  assert.deepEqual(
    merged.connections.map(c => [c.id, c.label, c.url || c.host, (c.token as any)?.value]),
    [
      ['local', 'Studio', undefined, undefined],
      ['web', 'Web added', 'http://web:1', 'w'],
      ['homelab', 'Same URL', 'http://10.0.0.5:9119/', 't'],
      ['web-added-2', 'Web added 2', 'http://other:1', 'o'],
      ['box', 'Box', 'box.lan', undefined]
    ]
  )
  assert.deepEqual([merged.primary, merged.launchMode, merged.lastUsed], ['homelab', 'last-used', 'web-added-2'])
  assert.equal(readJson(path.join(root, GATEWAY_CONNECTIONS_FILE)).revision, 4)

  // Removed in the shared list: a later launch must not bring it back from userData.
  store.write({ ...merged, connections: merged.connections.filter(c => c.id !== 'box') })
  const relaunched = storeFor(root, userData)
  assert.equal(
    relaunched.read().connections.some(c => c.id === 'box'),
    false
  )
})

test('a write made from a stale read keeps the other writer’s save', () => {
  const { root, userData } = scratch()
  const store = storeFor(root, userData)
  const seen = store.read()
  const doc = readJson(path.join(root, GATEWAY_CONNECTIONS_FILE))

  doc.connections.push({ id: 'web', kind: 'remote', label: 'Web', url: 'http://w:1', authMode: 'oauth' })
  doc.revision += 1
  writeJson(path.join(root, GATEWAY_CONNECTIONS_FILE), doc)

  store.write({ ...seen, connections: [...seen.connections, { id: 'desk', kind: 'ssh', label: 'Desk', host: 'd' }] })
  assert.deepEqual(
    readJson(path.join(root, GATEWAY_CONNECTIONS_FILE)).connections.map((c: any) => c.id),
    ['local', 'web', 'desk']
  )
})

test('the Python writer and this module share one file, one lock and each other’s saves', async () => {
  const { dir, root, userData } = scratch()
  const store = storeFor(root, userData)
  const seen = store.read()
  store.write({ ...seen, connections: [...seen.connections, { id: 'desk', kind: 'ssh', label: 'Desk', host: 'd' }] })

  const env = { HERMES_HOME: root, HOME: dir, PATH: process.env.PATH, PYTHONPATH: repo }

  const script = [
    'from hermes_cli import gateway_connections as g',
    'g.save_connection({"kind": "remote", "label": "From web", "url": "http://web:1", "token": "t"})',
    'print(",".join(c["id"] for c in g.read_registry()["connections"]))'
  ].join('\n')

  const { stdout } = await promisify(execFile)(python, ['-c', script], { env })
  assert.equal(stdout.trim(), 'local,desk,from-web')
  assert.deepEqual(
    store.read().connections.map(c => c.id),
    ['local', 'desk', 'from-web']
  )
})

test('another writer’s save replays as the same per-connection pushes a local save sends', () => {
  const before = normalizeRegistry({
    connections: [
      { id: 'a', kind: 'remote', label: 'A', url: 'http://a' },
      { id: 'b', kind: 'remote', label: 'B', url: 'http://b' },
      { id: 'c', kind: 'remote', label: 'C', url: 'http://c' }
    ]
  })

  const after = normalizeRegistry({
    connections: [
      { id: 'a', kind: 'remote', label: 'A renamed', url: 'http://a' },
      { id: 'b', kind: 'remote', label: 'B', url: 'http://b2' },
      { id: 'd', kind: 'ssh', label: 'D', host: 'd' }
    ]
  })

  assert.deepEqual(registryChanges(before, after), [
    { connectionId: 'a', reason: 'saved' },
    { connectionId: 'b', reason: 'updated' },
    { connectionId: 'c', reason: 'removed' },
    { connectionId: 'd', reason: 'saved' }
  ])
  assert.deepEqual(registryChanges(after, { ...after, primary: 'd' }), [{ connectionId: 'd', reason: 'saved' }])
})

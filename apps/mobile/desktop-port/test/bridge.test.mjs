import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { resolve, dirname } from 'node:path'
import vm from 'node:vm'

// Load the real shim (as shipped) inside a minimal headless environment. A
// single chainable Proxy absorbs every DOM call the shim's bootstrap makes
// (createElement/appendChild/getElementById/querySelector/addEventListener/…),
// so we exercise the actual window.hermesDesktop bridge without a browser and
// without modifying the shim.
const here = dirname(fileURLToPath(import.meta.url))
const SHIM_SRC = readFileSync(resolve(here, '../shim/hermes-web-shim.js'), 'utf8')

function loadBridge({ config = null, fetchImpl } = {}) {
  const store = new Map()
  if (config) store.set('hermes.remoteGateway', JSON.stringify(config))

  const localStorage = {
    getItem: (k) => (store.has(k) ? store.get(k) : null),
    setItem: (k, v) => store.set(k, String(v)),
    removeItem: (k) => store.delete(k),
  }

  // Chainable no-op node: any property access or call returns the node itself,
  // so arbitrarily deep DOM chains resolve without throwing.
  const domNode = new Proxy(function () {}, {
    get: () => domNode,
    apply: () => domNode,
    set: () => true,
  })

  const win = {
    localStorage,
    location: { href: 'capacitor://localhost/', origin: 'capacitor://localhost', reload() {} },
    addEventListener() {},
    removeEventListener() {},
    dispatchEvent() {
      return true
    },
    matchMedia: () => ({ matches: false, addEventListener() {}, removeEventListener() {} }),
    navigator: { userAgent: 'test' },
  }
  win.window = win

  const ctx = {
    window: win,
    document: domNode,
    location: win.location,
    navigator: win.navigator,
    localStorage,
    MutationObserver: class {
      observe() {}
      disconnect() {}
    },
    CustomEvent: class {
      constructor(type, opts) {
        this.type = type
        Object.assign(this, opts)
      }
    },
    WebSocket: class {
      addEventListener() {}
      close() {}
    },
    AbortSignal: { timeout: () => ({}) },
    URL,
    URLSearchParams,
    fetch: fetchImpl || (async () => {
      throw new Error('fetch was called unexpectedly')
    }),
    setTimeout: () => 0,
    clearTimeout: () => {},
    console,
  }
  vm.createContext(ctx)
  vm.runInContext(SHIM_SRC, ctx, { filename: 'hermes-web-shim.js' })
  return win.hermesDesktop
}

const CONFIG = { url: 'https://gateway.example', token: 'tok' }
const okFetch = async () => ({
  ok: true,
  status: 200,
  statusText: 'OK',
  headers: { get: () => 'application/json' },
  text: async () => '{"ok":true}',
})

test('the bridge loads headlessly and exposes the expected surface', () => {
  const d = loadBridge({ config: CONFIG })
  assert.equal(typeof d.api, 'function')
  assert.equal(typeof d.getConnection, 'function')
  assert.equal(typeof d.profile.set, 'function')
})

test('profile.set rejects a named profile (switching is disabled)', async () => {
  const d = loadBridge({ config: CONFIG })
  await assert.rejects(() => d.profile.set('work'), /single remote gateway/i)
})

test('profile.set accepts the primary profile (default / null / empty)', async () => {
  const d = loadBridge({ config: CONFIG })
  // Field checks (not deepEqual) — results cross the vm realm boundary.
  assert.equal((await d.profile.set('default')).profile, null)
  assert.equal((await d.profile.set(null)).profile, null)
  assert.equal((await d.profile.set('')).profile, null)
})

test('api throws for a foreign profile BEFORE issuing any request', async () => {
  let called = false
  const d = loadBridge({
    config: CONFIG,
    fetchImpl: async () => {
      called = true
      return okFetch()
    },
  })
  await assert.rejects(() => d.api({ path: '/api/git/status', profile: 'work' }), /single remote gateway/i)
  assert.equal(called, false, 'must not route a foreign profile to the unscoped backend')
})

test('api proceeds for the primary profile and parses JSON', async () => {
  const d = loadBridge({ config: CONFIG, fetchImpl: okFetch })
  assert.equal((await d.api({ path: '/api/status', profile: 'default' })).ok, true)
  assert.equal((await d.api({ path: '/api/status' })).ok, true)
})

test('getConnection is scoped to the primary profile', async () => {
  const d = loadBridge({ config: CONFIG })
  const conn = await d.getConnection()
  assert.equal(conn.mode, 'remote')
  assert.equal(conn.authMode, 'token')
  assert.equal(conn.profile, null)
  await assert.rejects(() => d.getConnection('work'), /single remote gateway/i)
})

test('getGatewayWsUrl and touchBackend reject a foreign profile', async () => {
  const d = loadBridge({ config: CONFIG })
  await assert.rejects(() => d.getGatewayWsUrl('work'), /single remote gateway/i)
  await assert.rejects(() => d.touchBackend('work'), /single remote gateway/i)
  assert.equal((await d.touchBackend()).ok, true)
})

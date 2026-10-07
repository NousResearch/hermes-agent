import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

import { test } from 'vitest'

import { applyWebviewHardening, hardenedWebviewPreferences, shouldBlockWebviewAttach } from './webview-hardening'

const mainSource = () => readFileSync(fileURLToPath(new URL('./main.ts', import.meta.url)), 'utf8')

test('hardenedWebviewPreferences strips a guest-requested nodeIntegration', () => {
  const hardened = hardenedWebviewPreferences({ nodeIntegration: true })

  assert.equal(hardened.nodeIntegration, false)
})

test('hardenedWebviewPreferences strips every Node-bearing escalation key at once', () => {
  const hardened = hardenedWebviewPreferences({
    contextIsolation: false,
    nodeIntegration: true,
    nodeIntegrationInSubFrames: true,
    nodeIntegrationInWorker: true,
    sandbox: false,
    webSecurity: false
  })

  assert.deepEqual(
    {
      contextIsolation: hardened.contextIsolation,
      nodeIntegration: hardened.nodeIntegration,
      nodeIntegrationInSubFrames: hardened.nodeIntegrationInSubFrames,
      nodeIntegrationInWorker: hardened.nodeIntegrationInWorker,
      sandbox: hardened.sandbox,
      webSecurity: hardened.webSecurity
    },
    {
      contextIsolation: true,
      nodeIntegration: false,
      nodeIntegrationInSubFrames: false,
      nodeIntegrationInWorker: false,
      sandbox: true,
      webSecurity: true
    }
  )
})

test('hardenedWebviewPreferences drops a guest preload instead of forwarding it', () => {
  const hardened = hardenedWebviewPreferences({ preload: '/tmp/evil-preload.js' })

  assert.equal('preload' in hardened, false)
})

test('hardenedWebviewPreferences refuses to let a guest re-enable the webview tag', () => {
  const hardened = hardenedWebviewPreferences({ webviewTag: true })

  assert.equal(hardened.webviewTag, false)
})

test('hardenedWebviewPreferences preserves unrelated preferences the preview pane relies on', () => {
  const hardened = hardenedWebviewPreferences({ partition: 'persist:hermes-preview', transparent: true })

  assert.equal(hardened.partition, 'persist:hermes-preview')
  assert.equal(hardened.transparent, true)
})

test('hardenedWebviewPreferences is safe on an empty request', () => {
  const hardened = hardenedWebviewPreferences()

  assert.equal(hardened.nodeIntegration, false)
  assert.equal(hardened.contextIsolation, true)
  assert.equal(hardened.sandbox, true)
})

test('shouldBlockWebviewAttach refuses a guest that brings its own preload', () => {
  assert.equal(shouldBlockWebviewAttach({ preload: '/tmp/evil-preload.js' }), true)
})

test('shouldBlockWebviewAttach allows the ordinary preview guest', () => {
  assert.equal(shouldBlockWebviewAttach({ src: 'https://example.com/' } as { preload?: unknown }), false)
  assert.equal(shouldBlockWebviewAttach({}), false)
  assert.equal(shouldBlockWebviewAttach({ preload: '' }), false)
})

// Electron honours whatever is left on the live object, so a delete-only key
// (preload) must really be gone — Object.assign alone would have kept it.
test('applyWebviewHardening deletes a guest preload from the live object', () => {
  const live: Record<string, unknown> = { nodeIntegration: true, preload: '/tmp/evil-preload.js' }

  applyWebviewHardening(live)

  assert.equal('preload' in live, false)
  assert.equal(live.nodeIntegration, false)
})

test('applyWebviewHardening mutates in place and returns the same object', () => {
  const live: Record<string, unknown> = { partition: 'persist:hermes-preview' }

  assert.equal(applyWebviewHardening(live), live)
  assert.equal(live.partition, 'persist:hermes-preview')
  assert.equal(live.sandbox, true)
})

// The pure helper only protects anything if main.ts actually subscribes. This is
// the regression that would otherwise rot silently: someone deletes the wiring,
// every unit test above still passes, and webviewTag is unguarded again.
test('main.ts wires the hardening into will-attach-webview', () => {
  const source = mainSource()

  assert.match(source, /will-attach-webview/)
  assert.match(source, /applyWebviewHardening/)
  assert.match(source, /shouldBlockWebviewAttach/)
})

/**
 * Unit tests for the `<webview>` window-open policy.
 *
 * The behaviour these lock down was verified against the pinned Electron
 * (40.10.2) with a live `<webview>`: without `allowpopups` the guest's
 * `setWindowOpenHandler` is never called, and with it the handler receives the
 * `target="_blank"` URL and an in-handler `loadURL` navigates the guest while
 * `deny` keeps any real window from being created.
 */

import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'

import { describe, test } from 'vitest'

import { decideWebviewWindowOpen, installWebviewWindowOpenPolicy } from './webview-window-open'

type Handler = (details: { url: string }) => { action: 'deny' }

function makeGuest({ loadRejects = false }: { loadRejects?: boolean } = {}) {
  const loaded: string[] = []
  let destroyed = false
  let handler: Handler | null = null

  return {
    loaded,
    destroy() {
      destroyed = true
    },
    /** Fire the installed handler the way Electron would. */
    requestWindow(url: string) {
      assert.ok(handler, 'no window-open handler installed on the guest')

      return handler({ url })
    },
    isDestroyed: () => destroyed,
    loadURL(url: string) {
      loaded.push(url)

      return loadRejects ? Promise.reject(new Error('ERR_ABORTED')) : Promise.resolve()
    },
    setWindowOpenHandler(next: Handler) {
      handler = next
    }
  }
}

function makeEmbedder() {
  const emitter = new EventEmitter()

  return {
    attach(guest: unknown) {
      emitter.emit('did-attach-webview', {}, guest)
    },
    listenerCount: () => emitter.listenerCount('did-attach-webview'),
    on: emitter.on.bind(emitter),
    off: emitter.off.bind(emitter)
  }
}

function install(embedder: ReturnType<typeof makeEmbedder>) {
  const external: string[] = []
  const logs: string[] = []

  const uninstall = installWebviewWindowOpenPolicy(embedder as never, {
    log: message => logs.push(message),
    openExternal: url => external.push(url)
  })

  return { external, logs, uninstall }
}

describe('decideWebviewWindowOpen', () => {
  test('web URLs navigate the requesting guest in place', () => {
    assert.deepEqual(decideWebviewWindowOpen('https://uzum.uz/ru/product/1'), {
      action: 'navigate',
      url: 'https://uzum.uz/ru/product/1'
    })
    assert.equal(decideWebviewWindowOpen('http://localhost:5173/page').action, 'navigate')
  })

  test('mailto goes to the OS handler', () => {
    assert.deepEqual(decideWebviewWindowOpen('mailto:a@b.co'), { action: 'external', url: 'mailto:a@b.co' })
  })

  test('file URLs are dropped, never forwarded to the OS', () => {
    // openExternalUrl() would hand a file: URL to shell.openPath; a previewed
    // remote page must not be able to reach the local filesystem that way.
    assert.deepEqual(decideWebviewWindowOpen('file:///etc/passwd'), { action: 'block' })
  })

  test('non-web schemes, about:blank and junk are dropped', () => {
    for (const url of ['about:blank', 'javascript:alert(1)', 'data:text/html,<b>x', 'chrome://settings', 'not a url', '', null, undefined]) {
      assert.equal(decideWebviewWindowOpen(url).action, 'block', `expected ${String(url)} to be blocked`)
    }
  })
})

describe('installWebviewWindowOpenPolicy', () => {
  test('a target=_blank click navigates the guest and denies the popup', () => {
    const embedder = makeEmbedder()

    install(embedder)

    const guest = makeGuest()

    embedder.attach(guest)

    assert.deepEqual(guest.requestWindow('https://example.com/target-page'), { action: 'deny' })
    assert.deepEqual(guest.loaded, ['https://example.com/target-page'])
  })

  test('mailto is handed to openExternal and never loaded in the pane', () => {
    const embedder = makeEmbedder()
    const { external } = install(embedder)
    const guest = makeGuest()

    embedder.attach(guest)
    guest.requestWindow('mailto:sales@example.com')

    assert.deepEqual(external, ['mailto:sales@example.com'])
    assert.deepEqual(guest.loaded, [])
  })

  test('blocked schemes touch neither the pane nor the OS', () => {
    const embedder = makeEmbedder()
    const { external } = install(embedder)
    const guest = makeGuest()

    embedder.attach(guest)

    assert.deepEqual(guest.requestWindow('file:///C:/Windows/System32/calc.exe'), { action: 'deny' })
    assert.deepEqual(guest.loaded, [])
    assert.deepEqual(external, [])
  })

  test('a destroyed guest is not navigated', () => {
    const embedder = makeEmbedder()

    install(embedder)

    const guest = makeGuest()

    embedder.attach(guest)
    guest.destroy()

    assert.deepEqual(guest.requestWindow('https://example.com/'), { action: 'deny' })
    assert.deepEqual(guest.loaded, [])
  })

  test('a rejected navigation is logged, not thrown', async () => {
    const embedder = makeEmbedder()
    const { logs } = install(embedder)
    const guest = makeGuest({ loadRejects: true })

    embedder.attach(guest)
    guest.requestWindow('https://example.com/')

    await new Promise(resolve => setTimeout(resolve, 0))

    assert.equal(logs.length, 1)
    assert.match(logs[0], /ERR_ABORTED/)
  })

  test('every attached guest gets the policy, and uninstall stops that', () => {
    const embedder = makeEmbedder()
    const { uninstall } = install(embedder)

    const first = makeGuest()
    const second = makeGuest()

    embedder.attach(first)
    embedder.attach(second)
    first.requestWindow('https://example.com/one')
    second.requestWindow('https://example.com/two')

    assert.deepEqual(first.loaded, ['https://example.com/one'])
    assert.deepEqual(second.loaded, ['https://example.com/two'])

    uninstall()

    assert.equal(embedder.listenerCount(), 0)
  })

  test('a missing embedder is a no-op', () => {
    assert.doesNotThrow(() => installWebviewWindowOpenPolicy(null, { openExternal: () => {} })())
  })
})

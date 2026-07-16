import assert from 'node:assert/strict'

import type { Session } from 'electron'
import { test } from 'vitest'

import {
  applyBrowserWebviewPolicy,
  BROWSER_WEBVIEW_PARTITION,
  browserGuestPopupPolicy,
  createBrowserCaptureCache,
  installBrowserGuestPermissionPolicy,
  isBrowserCapturePngDestination,
  isBrowserGuestNavigationAllowed,
  isBrowserWebviewSourceAllowed,
  PREVIEW_WEBVIEW_PARTITION,
  sanitizeBrowserCaptureFilename
} from './browser-webview'

test('webview source policy is partition-specific and rejects executable or remote-local sources', () => {
  assert.equal(isBrowserWebviewSourceAllowed(BROWSER_WEBVIEW_PARTITION, 'https://example.test/page'), true)
  assert.equal(isBrowserWebviewSourceAllowed(BROWSER_WEBVIEW_PARTITION, 'http://127.0.0.1:4173/fixture.html'), true)
  assert.equal(isBrowserWebviewSourceAllowed(BROWSER_WEBVIEW_PARTITION, 'file:///tmp/fixture.html'), false)
  assert.equal(isBrowserWebviewSourceAllowed(PREVIEW_WEBVIEW_PARTITION, 'file:///tmp/fixture.html'), true)
  assert.equal(isBrowserWebviewSourceAllowed(PREVIEW_WEBVIEW_PARTITION, 'file://server/share/fixture.html'), false)
  assert.equal(isBrowserWebviewSourceAllowed(PREVIEW_WEBVIEW_PARTITION, 'file:\\\\server\\share\\fixture.html'), false)
  assert.equal(isBrowserWebviewSourceAllowed(BROWSER_WEBVIEW_PARTITION, 'data:image/png;base64,AA=='), true)
  assert.equal(
    isBrowserWebviewSourceAllowed(BROWSER_WEBVIEW_PARTITION, 'data:image/svg+xml,%3Csvg%3E%3C/svg%3E'),
    false
  )
  assert.equal(
    isBrowserWebviewSourceAllowed(BROWSER_WEBVIEW_PARTITION, 'data:text/html;base64,PGgxPkhlbGxvPC9oMT4='),
    false
  )
  assert.equal(isBrowserWebviewSourceAllowed(BROWSER_WEBVIEW_PARTITION, 'javascript:alert(1)'), false)
  assert.equal(isBrowserWebviewSourceAllowed(BROWSER_WEBVIEW_PARTITION, 'blob:https://example.test/id'), false)

  const preferences: Record<string, unknown> = {
    preload: '/tmp/hostile-preload.js',
    contextIsolation: false,
    sandbox: false,
    webSecurity: false,
    nodeIntegration: true,
    nodeIntegrationInSubFrames: true
  }

  const params = {
    partition: BROWSER_WEBVIEW_PARTITION,
    src: 'https://example.test',
    preload: '/tmp/hostile-preload.js'
  }

  assert.equal(applyBrowserWebviewPolicy(params, preferences), true)
  assert.deepEqual(preferences, {
    contextIsolation: true,
    sandbox: true,
    webSecurity: true,
    nodeIntegration: false,
    nodeIntegrationInSubFrames: false
  })
  assert.deepEqual(params, {
    partition: BROWSER_WEBVIEW_PARTITION,
    src: 'https://example.test'
  })
  assert.equal(
    applyBrowserWebviewPolicy({ partition: PREVIEW_WEBVIEW_PARTITION, src: 'file:///tmp/fixture.html' }, {}),
    true
  )
  assert.equal(
    applyBrowserWebviewPolicy({ partition: 'persist:other', src: 'https://example.test' }, preferences),
    false
  )
  assert.equal(
    applyBrowserWebviewPolicy({ partition: BROWSER_WEBVIEW_PARTITION, src: 'javascript:alert(1)' }, preferences),
    false
  )
  assert.equal(
    applyBrowserWebviewPolicy({ partition: BROWSER_WEBVIEW_PARTITION, src: 'file:///tmp/fixture.html' }, preferences),
    false
  )
})

test('browser guests deny popups and retain partition-scoped navigation policy', () => {
  assert.equal(isBrowserGuestNavigationAllowed(BROWSER_WEBVIEW_PARTITION, 'https://example.test/next'), true)
  assert.equal(isBrowserGuestNavigationAllowed(BROWSER_WEBVIEW_PARTITION, 'file:///tmp/fixture.html'), false)
  assert.equal(isBrowserGuestNavigationAllowed(PREVIEW_WEBVIEW_PARTITION, 'file:///tmp/fixture.html'), true)
  assert.equal(isBrowserGuestNavigationAllowed(BROWSER_WEBVIEW_PARTITION, 'data:image/webp;base64,AA=='), true)
  assert.equal(isBrowserGuestNavigationAllowed(BROWSER_WEBVIEW_PARTITION, 'mailto:hello@example.test'), false)
  assert.equal(isBrowserGuestNavigationAllowed(BROWSER_WEBVIEW_PARTITION, 'javascript:alert(1)'), false)
  assert.deepEqual(browserGuestPopupPolicy('https://example.test/new-window'), { action: 'deny' })
  assert.deepEqual(browserGuestPopupPolicy('javascript:alert(1)'), { action: 'deny' })
})

test('browser guest permission policy denies every request and permission check', () => {
  type PermissionRequestHandler = Exclude<Parameters<Session['setPermissionRequestHandler']>[0], null>
  type PermissionCheckHandler = Exclude<Parameters<Session['setPermissionCheckHandler']>[0], null>

  let requestHandler: PermissionRequestHandler | undefined
  let checkHandler: PermissionCheckHandler | undefined

  const browserSession = {
    setPermissionRequestHandler(handler: PermissionRequestHandler | null) {
      requestHandler = handler ?? undefined
    },
    setPermissionCheckHandler(handler: PermissionCheckHandler | null) {
      checkHandler = handler ?? undefined
    }
  }

  installBrowserGuestPermissionPolicy(browserSession)

  assert.ok(requestHandler)
  assert.ok(checkHandler)

  for (const { label, permission, requestDetails, checkDetails } of [
    {
      label: 'camera',
      permission: 'media',
      requestDetails: { mediaTypes: ['video'] },
      checkDetails: { mediaType: 'video' }
    },
    {
      label: 'microphone/media',
      permission: 'media',
      requestDetails: { mediaTypes: ['audio'] },
      checkDetails: { mediaType: 'audio' }
    },
    {
      label: 'geolocation',
      permission: 'geolocation',
      requestDetails: {},
      checkDetails: {}
    },
    {
      label: 'notifications',
      permission: 'notifications',
      requestDetails: {},
      checkDetails: {}
    },
    {
      label: 'clipboard',
      permission: 'clipboard-read',
      requestDetails: {},
      checkDetails: {}
    },
    {
      label: 'fullscreen',
      permission: 'fullscreen',
      requestDetails: {},
      checkDetails: {}
    },
    {
      label: 'MIDI',
      permission: 'midi',
      requestDetails: {},
      checkDetails: {}
    },
    {
      label: 'unknown permission',
      permission: 'unknown-permission',
      requestDetails: {},
      checkDetails: {}
    }
  ]) {
    let granted: boolean | undefined
    requestHandler(
      null as never,
      permission as never,
      value => {
        granted = value
      },
      requestDetails as never
    )
    assert.equal(granted, false, `${label} request is denied`)
    assert.equal(
      checkHandler(null as never, permission as never, 'https://example.test', checkDetails as never),
      false,
      `${label} check is denied`
    )
  }
})

test('browser capture cache scopes entries to an owner and expires bounded entries', () => {
  let currentTime = 1_000
  let nextId = 0

  const cache = createBrowserCaptureCache({
    maxEntries: 2,
    ttlMs: 100,
    now: () => currentTime,
    createId: () => `capture-${++nextId}`
  })

  const png = new Uint8Array([137, 80, 78, 71])
  const first = cache.put({ ownerId: 7, png, width: 320, height: 200 })

  assert.deepEqual(cache.get(first, 7), {
    captureId: first,
    ownerId: 7,
    png,
    width: 320,
    height: 200,
    createdAt: 1_000
  })
  assert.equal(cache.get(first, 8), null)
  assert.equal(cache.remove(first, 8), false)

  cache.put({ ownerId: 7, png, width: 1, height: 1 })
  cache.put({ ownerId: 7, png, width: 2, height: 2 })
  assert.equal(cache.get(first, 7), null)
  assert.equal(cache.size, 2)

  currentTime = 1_100
  assert.equal(cache.get('capture-3', 7), null)
  assert.equal(cache.size, 0)
})

test('browser capture cache is byte-bounded and rejects oversized captures', () => {
  let nextId = 0

  const cache = createBrowserCaptureCache({
    maxEntries: 3,
    maxBytes: 5,
    createId: () => `capture-${++nextId}`
  })

  const first = cache.put({ ownerId: 7, png: new Uint8Array(3), width: 1, height: 1 })
  const second = cache.put({ ownerId: 7, png: new Uint8Array(3), width: 1, height: 1 })

  assert.equal(first, 'capture-1')
  assert.equal(second, 'capture-2')
  assert.equal(cache.get(first, 7), null)
  assert.equal(cache.bytes, 3)
  assert.equal(cache.put({ ownerId: 7, png: new Uint8Array(6), width: 1, height: 1 }), null)
  assert.equal(cache.bytes, 3)
})

test('browser capture cache expires while idle and clears destroyed-window owners', () => {
  let currentTime = 1_000
  let expiry: (() => void) | undefined

  const cache = createBrowserCaptureCache({
    ttlMs: 100,
    now: () => currentTime,
    setTimeoutFn: callback => {
      expiry = callback

      return { unref() {} } as ReturnType<typeof setTimeout>
    },
    clearTimeoutFn: () => {
      expiry = undefined
    }
  })

  cache.put({ ownerId: 7, png: new Uint8Array(1), width: 1, height: 1 })
  currentTime = 1_100
  expiry?.()
  assert.equal(cache.size, 0)

  const ownerCapture = cache.put({ ownerId: 7, png: new Uint8Array(1), width: 1, height: 1 })
  const otherCapture = cache.put({ ownerId: 8, png: new Uint8Array(1), width: 1, height: 1 })
  cache.removeOwner(7)

  assert.equal(cache.get(ownerCapture, 7), null)
  assert.notEqual(cache.get(otherCapture, 8), null)
})

test('browser capture filenames and destinations remain PNG-only', () => {
  assert.equal(sanitizeBrowserCaptureFilename('../Quarterly report.png'), 'Quarterly report.png')
  assert.equal(sanitizeBrowserCaptureFilename('..\\unsafe:name'), 'unsafe-name.png')
  assert.equal(sanitizeBrowserCaptureFilename(''), 'browser-capture.png')
  assert.equal(isBrowserCapturePngDestination('/tmp/capture.png'), true)
  assert.equal(isBrowserCapturePngDestination('/tmp/capture.PNG'), true)
  assert.equal(isBrowserCapturePngDestination('/tmp/capture.png '), false)
  assert.equal(isBrowserCapturePngDestination(' /tmp/capture.png'), false)
  assert.equal(isBrowserCapturePngDestination('/tmp/capture.png/child'), false)
  assert.equal(isBrowserCapturePngDestination('/tmp/capture.jpg'), false)
})

test('main installs deny-by-default permission policy for both webview partitions', async () => {
  const source = await import('node:fs/promises').then(fs => fs.readFile(new URL('./main.ts', import.meta.url), 'utf8'))

  for (const partition of ['BROWSER_WEBVIEW_PARTITION', 'PREVIEW_WEBVIEW_PARTITION']) {
    assert.match(
      source,
      new RegExp(`installBrowserGuestPermissionPolicy\\(session\\.fromPartition\\(${partition}\\)\\)`)
    )
  }
})

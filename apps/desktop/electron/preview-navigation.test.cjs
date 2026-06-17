const test = require('node:test')
const assert = require('node:assert/strict')

const {
  guardPreviewWebContents,
  isAllowedPreviewExternalOpen,
  isAllowedPreviewNavigation
} = require('./preview-navigation.cjs')

function fakeWebContents() {
  const listeners = new Map()
  let windowOpenHandler = null

  return {
    emit(name, ...args) {
      listeners.get(name)?.(...args)
    },
    on(name, listener) {
      listeners.set(name, listener)
    },
    setWindowOpenHandler(handler) {
      windowOpenHandler = handler
    },
    windowOpen(url) {
      return windowOpenHandler({ url })
    }
  }
}

test('isAllowedPreviewNavigation allows preview-safe schemes', () => {
  assert.equal(isAllowedPreviewNavigation('https://example.com/page'), true)
  assert.equal(isAllowedPreviewNavigation('http://localhost:5173'), true)
  assert.equal(isAllowedPreviewNavigation('file:///tmp/demo.html'), true)
  assert.equal(isAllowedPreviewNavigation('hermes-media://stream/%2Ftmp%2Fdemo.mp4'), true)
})

test('isAllowedPreviewNavigation rejects custom and executable schemes', () => {
  assert.equal(isAllowedPreviewNavigation('bitbrowser://open?url=https%3A%2F%2Fexample.com'), false)
  assert.equal(isAllowedPreviewNavigation('intent://scan/#Intent;scheme=zxing;end'), false)
  assert.equal(isAllowedPreviewNavigation('javascript:alert(1)'), false)
  assert.equal(isAllowedPreviewNavigation('data:text/html,<script>alert(1)</script>'), false)
  assert.equal(isAllowedPreviewNavigation(''), false)
})

test('isAllowedPreviewExternalOpen only allows web URLs', () => {
  assert.equal(isAllowedPreviewExternalOpen('https://example.com/page'), true)
  assert.equal(isAllowedPreviewExternalOpen('http://example.com/page'), true)
  assert.equal(isAllowedPreviewExternalOpen('file:///tmp/demo.html'), false)
  assert.equal(isAllowedPreviewExternalOpen('hermes-media://stream/%2Ftmp%2Fdemo.mp4'), false)
  assert.equal(isAllowedPreviewExternalOpen('bitbrowser://open'), false)
})

test('guardPreviewWebContents blocks custom-protocol guest navigation', () => {
  const webContents = fakeWebContents()
  let prevented = false

  guardPreviewWebContents(webContents)
  webContents.emit(
    'will-navigate',
    { preventDefault: () => (prevented = true) },
    'bitbrowser://open?url=https%3A%2F%2Fexample.com'
  )

  assert.equal(prevented, true)
})

test('guardPreviewWebContents blocks custom-protocol guest redirects and frame navigations', () => {
  const webContents = fakeWebContents()
  const blocked = []

  guardPreviewWebContents(webContents)
  webContents.emit('will-redirect', { preventDefault: () => blocked.push('redirect') }, 'intent://scan')
  webContents.emit('will-frame-navigate', { preventDefault: () => blocked.push('frame') }, 'bitbrowser://open')

  assert.deepEqual(blocked, ['redirect', 'frame'])
})

test('guardPreviewWebContents leaves preview-safe guest navigation alone', () => {
  const webContents = fakeWebContents()
  let prevented = false

  guardPreviewWebContents(webContents)
  webContents.emit('will-navigate', { preventDefault: () => (prevented = true) }, 'https://example.com/next')

  assert.equal(prevented, false)
})

test('guardPreviewWebContents denies webview new windows and opens http links externally', () => {
  const webContents = fakeWebContents()
  const opened = []

  guardPreviewWebContents(webContents, {
    openExternalUrl(url) {
      opened.push(url)
    }
  })

  assert.deepEqual(webContents.windowOpen('https://example.com/page'), { action: 'deny' })
  assert.deepEqual(webContents.windowOpen('bitbrowser://open'), { action: 'deny' })
  assert.deepEqual(webContents.windowOpen('file:///tmp/demo.html'), { action: 'deny' })
  assert.deepEqual(opened, ['https://example.com/page'])
})

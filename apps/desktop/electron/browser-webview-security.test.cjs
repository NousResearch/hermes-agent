const assert = require('node:assert/strict')
const test = require('node:test')

const {
  BROWSER_WEBVIEW_PARTITION_PREFIX,
  browserWebviewPermissionAllowed,
  installBrowserWebviewSecurity,
  isAllowedBrowserWebviewSrc,
  isHermesBrowserPartition,
  sanitizeBrowserWebPreferences
} = require('./browser-webview-security.cjs')

test('sanitizeBrowserWebPreferences forces sandboxed isolated non-node webviews and strips preload', () => {
  const params = {
    preload: '/tmp/evil.js',
    sandbox: false,
    nodeIntegration: true,
    contextIsolation: false,
    javascript: false
  }

  sanitizeBrowserWebPreferences(params)

  assert.equal(params.preload, undefined)
  assert.equal(params.sandbox, true)
  assert.equal(params.nodeIntegration, false)
  assert.equal(params.contextIsolation, true)
  assert.equal(params.javascript, false, 'unrelated preferences are preserved')
})

test('isAllowedBrowserWebviewSrc accepts browser-safe schemes and rejects privileged schemes', () => {
  assert.equal(isAllowedBrowserWebviewSrc('about:blank'), true)
  assert.equal(isAllowedBrowserWebviewSrc('https://example.com'), true)
  assert.equal(isAllowedBrowserWebviewSrc('http://127.0.0.1:5173'), true)

  assert.equal(isAllowedBrowserWebviewSrc('file:///etc/passwd'), false)
  assert.equal(isAllowedBrowserWebviewSrc('javascript:alert(1)'), false)
  assert.equal(isAllowedBrowserWebviewSrc('data:text/html,owned'), false)
  assert.equal(isAllowedBrowserWebviewSrc('chrome://gpu'), false)
  assert.equal(isAllowedBrowserWebviewSrc('not a url'), false)
})

test('browserWebviewPermissionAllowed is deny-by-default', () => {
  for (const permission of ['media', 'geolocation', 'notifications', 'clipboard-read', 'fullscreen', 'unknown']) {
    assert.equal(browserWebviewPermissionAllowed(permission), false)
  }
})

test('isHermesBrowserPartition protects persistent and legacy Hermes browser namespaces only', () => {
  assert.equal(isHermesBrowserPartition(`${BROWSER_WEBVIEW_PARTITION_PREFIX}default:session-1`), true)
  assert.equal(isHermesBrowserPartition('hermes-browser:legacy'), true)
  assert.equal(isHermesBrowserPartition('persist:hermes-preview'), false)
  assert.equal(isHermesBrowserPartition('persist:evil'), false)
})

test('installBrowserWebviewSecurity clamps only Hermes browser webview partitions', () => {
  const app = fakeEventTarget()
  const defaultSession = fakeSession()
  const browserSession = fakeSession()
  const session = {
    defaultSession,
    fromPartition(partition) {
      assert.equal(partition, `${BROWSER_WEBVIEW_PARTITION_PREFIX}abc`)
      return browserSession
    }
  }

  installBrowserWebviewSecurity({ app, session })

  assert.equal(app.listeners['web-contents-created'].length, 1)
  const contents = fakeEventTarget()
  app.emit('web-contents-created', {}, contents)

  const attach = contents.listeners['will-attach-webview'][0]
  const ignored = fakeEvent()
  const ignoredParams = { partition: 'persist:hermes-preview', src: 'file:///preview.html' }
  attach(ignored, {}, ignoredParams)

  assert.equal(ignored.prevented, false)

  const denied = fakeEvent()
  const webPreferences = { preload: '/tmp/evil.js' }
  const params = { partition: `${BROWSER_WEBVIEW_PARTITION_PREFIX}abc`, src: 'file:///secret.txt' }
  attach(denied, webPreferences, params)

  assert.equal(denied.prevented, true)
  assert.equal(webPreferences.preload, undefined)
  assert.equal(webPreferences.sandbox, true)
  assert.equal(webPreferences.nodeIntegration, false)
  assert.equal(webPreferences.contextIsolation, true)

  const allowed = fakeEvent()
  const allowedParams = { partition: `${BROWSER_WEBVIEW_PARTITION_PREFIX}abc`, src: 'https://example.com' }
  attach(allowed, {}, allowedParams)

  assert.equal(allowed.prevented, false)
  assert.equal(browserSession.permissionRequestHandlerInstalled, true)
  assert.equal(browserSession.permissionCheckHandlerInstalled, true)
  assert.deepEqual(browserSession.webRequestFilter, { urls: ['<all_urls>'] })
  assert.deepEqual(browserSession.webRequestHandler({ url: 'file:///secret.txt' }), { cancel: true })
  assert.deepEqual(browserSession.webRequestHandler({ url: 'https://example.com/app.js' }), { cancel: false })

  const guest = fakeGuestContents()
  contents.emit('did-attach-webview', {}, guest)
  assert.equal(guest.windowOpenHandlerResult.action, 'deny')
})

function fakeEventTarget() {
  const listeners = {}

  return {
    listeners,
    on(name, handler) {
      listeners[name] ??= []
      listeners[name].push(handler)
    },
    emit(name, ...args) {
      for (const handler of listeners[name] ?? []) {
        handler(...args)
      }
    }
  }
}

function fakeEvent() {
  return {
    prevented: false,
    preventDefault() {
      this.prevented = true
    }
  }
}

function fakeSession() {
  const fake = {
    permissionRequestHandlerInstalled: false,
    permissionCheckHandlerInstalled: false,
    webRequestFilter: null,
    webRequestHandler: null,
    setPermissionRequestHandler(handler) {
      this.permissionRequestHandlerInstalled = true
      handler(null, 'media', allowed => {
        assert.equal(allowed, false)
      })
    },
    setPermissionCheckHandler(handler) {
      this.permissionCheckHandlerInstalled = true
      assert.equal(handler(null, 'media'), false)
    },
    webRequest: {
      onBeforeRequest(filter, handler) {
        fake.webRequestFilter = filter
        fake.webRequestHandler = details => {
          let result = null
          handler(details, response => {
            result = response
          })

          return result
        }
      }
    }
  }

  return fake
}

function fakeGuestContents() {
  return {
    setWindowOpenHandler(handler) {
      this.windowOpenHandlerResult = handler({ url: 'https://example.com/popup' })
    }
  }
}

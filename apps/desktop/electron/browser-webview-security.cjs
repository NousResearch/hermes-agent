// Security guardrails for the integrated BrowserPane <webview> surface.
// Kept Electron-free except for injected app/session objects so node --test can
// cover the dangerous decisions without booting Electron.

const BROWSER_WEBVIEW_PARTITION_PREFIX = 'persist:hermes-browser:'
const LEGACY_BROWSER_WEBVIEW_PARTITION_PREFIX = 'hermes-browser:'

const SAFE_BROWSER_PROTOCOLS = new Set(['about:', 'http:', 'https:'])
const securedSessions = new WeakSet()

function isHermesBrowserPartition(partition) {
  return (
    typeof partition === 'string' &&
    (partition.startsWith(BROWSER_WEBVIEW_PARTITION_PREFIX) ||
      partition.startsWith(LEGACY_BROWSER_WEBVIEW_PARTITION_PREFIX))
  )
}

function sanitizeBrowserWebPreferences(webPreferences) {
  if (!webPreferences || typeof webPreferences !== 'object') {
    return webPreferences
  }

  delete webPreferences.preload
  delete webPreferences.preloadURL
  webPreferences.contextIsolation = true
  webPreferences.nodeIntegration = false
  webPreferences.sandbox = true

  return webPreferences
}

function isAllowedBrowserWebviewSrc(src) {
  if (typeof src !== 'string' || !src.trim()) {
    return false
  }

  const value = src.trim()

  if (value === 'about:blank') {
    return true
  }

  let parsed

  try {
    parsed = new URL(value)
  } catch {
    return false
  }

  if (parsed.protocol === 'about:') {
    return parsed.href === 'about:blank'
  }

  return SAFE_BROWSER_PROTOCOLS.has(parsed.protocol)
}

function browserWebviewPermissionAllowed() {
  return false
}

function secureBrowserWebviewSession(browserSession) {
  if (!browserSession || securedSessions.has(browserSession)) {
    return
  }

  browserSession.setPermissionRequestHandler?.((_webContents, permission, callback) => {
    callback(browserWebviewPermissionAllowed(permission))
  })

  browserSession.setPermissionCheckHandler?.((_webContents, permission) => browserWebviewPermissionAllowed(permission))

  browserSession.webRequest?.onBeforeRequest?.({ urls: ['<all_urls>'] }, (details, callback) => {
    callback({ cancel: !isAllowedBrowserWebviewSrc(details?.url) })
  })

  securedSessions.add(browserSession)
}

function installBrowserWebviewSecurity({ app, session }) {
  if (!app || typeof app.on !== 'function' || !session) {
    return
  }

  app.on('web-contents-created', (_event, contents) => {
    contents?.on?.('will-attach-webview', (event, webPreferences = {}, params = {}) => {
      if (!isHermesBrowserPartition(params.partition)) {
        return
      }

      sanitizeBrowserWebPreferences(webPreferences)

      if (!isAllowedBrowserWebviewSrc(params.src)) {
        event?.preventDefault?.()

        return
      }

      const browserSession = session.fromPartition?.(params.partition)
      secureBrowserWebviewSession(browserSession)
    })

    contents?.on?.('did-attach-webview', (_event, guestContents) => {
      // Electron's did-attach-webview callback does not reliably include the
      // original <webview> partition/src params. Keep popup denial broad for
      // attached webviews as a conscious defense-in-depth posture until we have
      // a stable browser-partition discriminator at this lifecycle point.
      guestContents?.setWindowOpenHandler?.(() => ({ action: 'deny' }))
    })
  })
}

module.exports = {
  BROWSER_WEBVIEW_PARTITION_PREFIX,
  browserWebviewPermissionAllowed,
  installBrowserWebviewSecurity,
  isAllowedBrowserWebviewSrc,
  isHermesBrowserPartition,
  sanitizeBrowserWebPreferences,
  secureBrowserWebviewSession
}

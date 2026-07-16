import type { Session } from 'electron'

export const BROWSER_WEBVIEW_PARTITION = 'persist:hermes-browser'
export const PREVIEW_WEBVIEW_PARTITION = 'persist:hermes-preview'
export const BROWSER_CAPTURE_MAX_ENTRIES = 12
export const BROWSER_CAPTURE_TTL_MS = 5 * 60 * 1000
export const BROWSER_CAPTURE_MAX_BYTES = 32 * 1024 * 1024

interface BrowserWebviewParams {
  partition?: unknown
  src?: unknown
  preload?: unknown
}

interface BrowserWebviewPreferences {
  preload?: unknown
  contextIsolation?: unknown
  sandbox?: unknown
  webSecurity?: unknown
  nodeIntegration?: unknown
  nodeIntegrationInSubFrames?: unknown
}

interface BrowserCapture {
  ownerId: number
  png: Uint8Array
  width: number
  height: number
  createdAt: number
}

interface BrowserCaptureResult extends BrowserCapture {
  captureId: string
}

function parseUrl(rawUrl: unknown): URL | null {
  if (typeof rawUrl !== 'string' || !rawUrl.trim()) {
    return null
  }

  try {
    return new URL(rawUrl)
  } catch {
    return null
  }
}

export function isBrowserWebviewSourceAllowed(partition: unknown, rawUrl: unknown): boolean {
  const url = parseUrl(rawUrl)

  if ((partition !== BROWSER_WEBVIEW_PARTITION && partition !== PREVIEW_WEBVIEW_PARTITION) || !url) {
    return false
  }

  if (url.protocol === 'http:' || url.protocol === 'https:') {
    return true
  }

  if (url.protocol === 'file:') {
    return (
      partition === PREVIEW_WEBVIEW_PARTITION && !url.hostname && typeof rawUrl === 'string' && !rawUrl.includes('\\')
    )
  }

  return url.protocol === 'data:' && /^image\/(?:png|jpeg|jpg|webp|gif|avif|bmp)(?:;|,)/i.test(url.pathname)
}

export function applyBrowserWebviewPolicy(
  params: BrowserWebviewParams,
  preferences: BrowserWebviewPreferences
): boolean {
  if (!isBrowserWebviewSourceAllowed(params.partition, params.src)) {
    return false
  }

  delete params.preload
  delete preferences.preload
  preferences.contextIsolation = true
  preferences.sandbox = true
  preferences.webSecurity = true
  preferences.nodeIntegration = false
  preferences.nodeIntegrationInSubFrames = false

  return true
}

export function isBrowserGuestNavigationAllowed(partition: unknown, rawUrl: unknown): boolean {
  return isBrowserWebviewSourceAllowed(partition, rawUrl)
}

export function browserGuestPopupPolicy(_rawUrl: unknown): { action: 'deny' } {
  return { action: 'deny' }
}

export function installBrowserGuestPermissionPolicy(
  browserSession: Pick<Session, 'setPermissionRequestHandler' | 'setPermissionCheckHandler'>
): void {
  browserSession.setPermissionRequestHandler((_webContents, _permission, callback) => {
    callback(false)
  })
  browserSession.setPermissionCheckHandler(() => false)
}

export function sanitizeBrowserCaptureFilename(suggestedName: unknown): string {
  const fallback = 'browser-capture'
  const raw = typeof suggestedName === 'string' ? suggestedName.trim() : ''
  const basename = raw.split(/[\\/]/).pop() || ''

  const safeBasename = Array.from(basename, character =>
    character.charCodeAt(0) < 32 || '<>:"|?*'.includes(character) ? '-' : character
  ).join('')

  const stem = safeBasename
    .replace(/\.png$/i, '')
    .replace(/^\.+/, '')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 96)

  return `${stem || fallback}.png`
}

export function isBrowserCapturePngDestination(filePath: unknown): filePath is string {
  return typeof filePath === 'string' && filePath.trim() === filePath && /\.png$/i.test(filePath)
}

type BrowserCaptureTimeout = ReturnType<typeof setTimeout>
type BrowserCaptureSetTimeout = (callback: () => void, delay: number) => BrowserCaptureTimeout
type BrowserCaptureClearTimeout = (timeout: BrowserCaptureTimeout) => void

export function createBrowserCaptureCache({
  maxEntries = BROWSER_CAPTURE_MAX_ENTRIES,
  maxBytes = BROWSER_CAPTURE_MAX_BYTES,
  ttlMs = BROWSER_CAPTURE_TTL_MS,
  now = () => Date.now(),
  createId = () => crypto.randomUUID(),
  setTimeoutFn = setTimeout,
  clearTimeoutFn = clearTimeout
}: {
  maxEntries?: number
  maxBytes?: number
  ttlMs?: number
  now?: () => number
  createId?: () => string
  setTimeoutFn?: BrowserCaptureSetTimeout
  clearTimeoutFn?: BrowserCaptureClearTimeout
} = {}) {
  const captures = new Map<string, BrowserCapture>()
  let totalBytes = 0
  let expiryTimer: BrowserCaptureTimeout | undefined

  const boundedMaxEntries = Number.isFinite(maxEntries) ? Math.max(0, Math.floor(maxEntries)) : 0
  const boundedMaxBytes = Number.isFinite(maxBytes) ? Math.max(0, Math.floor(maxBytes)) : 0
  const boundedTtlMs = Number.isFinite(ttlMs) ? Math.max(0, ttlMs) : 0

  function deleteCapture(captureId: string): boolean {
    const capture = captures.get(captureId)

    if (!capture) {
      return false
    }

    totalBytes -= capture.png.byteLength
    captures.delete(captureId)

    return true
  }

  function scheduleExpiry() {
    if (expiryTimer !== undefined) {
      clearTimeoutFn(expiryTimer)
      expiryTimer = undefined
    }

    const oldest = captures.values().next().value as BrowserCapture | undefined

    if (!oldest) {
      return
    }

    const delay = Math.max(0, oldest.createdAt + boundedTtlMs - now())
    expiryTimer = setTimeoutFn(() => {
      expiryTimer = undefined
      prune()
    }, delay)
    expiryTimer.unref?.()
  }

  function prune() {
    const expiresBefore = now() - boundedTtlMs

    for (const [captureId, capture] of captures) {
      if (capture.createdAt <= expiresBefore) {
        deleteCapture(captureId)
      }
    }

    while (captures.size > boundedMaxEntries || totalBytes > boundedMaxBytes) {
      const oldest = captures.keys().next().value

      if (oldest === undefined) {
        break
      }

      deleteCapture(oldest)
    }

    scheduleExpiry()
  }

  function put({ ownerId, png, width, height }: Omit<BrowserCapture, 'createdAt'>): string | null {
    prune()

    if (png.byteLength > boundedMaxBytes || boundedMaxEntries === 0) {
      return null
    }

    while (captures.size >= boundedMaxEntries || totalBytes + png.byteLength > boundedMaxBytes) {
      const oldest = captures.keys().next().value

      if (oldest === undefined) {
        break
      }

      deleteCapture(oldest)
    }

    const captureId = createId()
    captures.set(captureId, { ownerId, png, width, height, createdAt: now() })
    totalBytes += png.byteLength
    scheduleExpiry()

    return captureId
  }

  function get(captureId: unknown, ownerId: number): BrowserCaptureResult | null {
    prune()

    if (typeof captureId !== 'string') {
      return null
    }

    const capture = captures.get(captureId)

    if (!capture || capture.ownerId !== ownerId) {
      return null
    }

    return { captureId, ...capture }
  }

  function remove(captureId: unknown, ownerId: number): boolean {
    const capture = get(captureId, ownerId)

    if (!capture) {
      return false
    }

    const removed = deleteCapture(capture.captureId)
    scheduleExpiry()

    return removed
  }

  function removeOwner(ownerId: number): void {
    for (const [captureId, capture] of captures) {
      if (capture.ownerId === ownerId) {
        deleteCapture(captureId)
      }
    }

    scheduleExpiry()
  }

  function clear(): void {
    captures.clear()
    totalBytes = 0
    scheduleExpiry()
  }

  return {
    clear,
    get,
    put,
    remove,
    removeOwner,
    prune,
    get size() {
      return captures.size
    },
    get bytes() {
      return totalBytes
    }
  }
}

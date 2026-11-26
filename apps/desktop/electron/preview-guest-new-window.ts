// A preview page that opens a new tab (`target="_blank"` link, `_blank` form
// POST, `window.open`) gets a new PREVIEW tab in the same chat and profile,
// never an Electron popup and never the system browser.
//
// The preview `<webview>` carries `allowpopups` only so Chromium hands these
// requests to `setWindowOpenHandler`; the handler installed here always answers
// `deny`, so no BrowserWindow is ever created (the GHSA-9f4c-93c8-jc8g popup
// contract still holds). The request is forwarded to the embedding renderer,
// which opens the tab next to the opener (same owner, same cookie jar).
//
// Rules (the renderer applies the same scheme rule again on its side):
// - http/https only; anything else is dropped.
// - A form POST travels with its body (raw form data only; a file upload part
//   is not replayed and the request is dropped).
// - At most POPUP_LIMIT requests per guest per POPUP_WINDOW_MS, so a page
//   cannot flood the strip with tabs.

export const PREVIEW_GUEST_NEW_WINDOW_CHANNEL = 'hermes:preview-guest-new-window'

export const POPUP_LIMIT = 5
export const POPUP_WINDOW_MS = 10_000

export interface PreviewGuestPost {
  body: Uint8Array
  /** The full `Content-Type` header value, boundary included. */
  contentType: string
}

export interface PreviewGuestNewWindow {
  /** Open behind the current tab (ctrl/middle-click) instead of fronting it. */
  background: boolean
  /** webContents id of the guest that asked; the pane matches it to its tab. */
  guestId: number
  post?: PreviewGuestPost
  url: string
}

/** The subset of Electron's `HandlerDetails` this module reads. */
export interface GuestWindowOpenDetails {
  disposition?: string
  postBody?: {
    boundary?: string
    contentType?: string
    data?: Array<{ bytes?: Uint8Array; filePath?: string; type?: string }>
  }
  url: string
}

export function isPreviewTabUrl(url: string): boolean {
  try {
    const { protocol } = new URL(url)

    return protocol === 'http:' || protocol === 'https:'
  } catch {
    return false
  }
}

function postFrom(details: GuestWindowOpenDetails): null | PreviewGuestPost | undefined {
  const post = details.postBody

  if (!post) {
    return undefined
  }

  const parts = post.data ?? []

  if (parts.some(part => part.type !== 'rawData' || !part.bytes)) {
    return null
  }

  const total = parts.reduce((sum, part) => sum + (part.bytes?.byteLength ?? 0), 0)
  const body = new Uint8Array(total)
  let offset = 0

  for (const part of parts) {
    body.set(part.bytes as Uint8Array, offset)
    offset += (part.bytes as Uint8Array).byteLength
  }

  const type = post.contentType || 'application/x-www-form-urlencoded'
  const contentType = post.boundary && !/boundary=/i.test(type) ? `${type}; boundary=${post.boundary}` : type

  return { body, contentType }
}

/** Per-guest sliding window: true when this request is still under the limit. */
export function createPopupLimiter(limit = POPUP_LIMIT, windowMs = POPUP_WINDOW_MS) {
  const seen = new Map<number, number[]>()

  return (guestId: number, now: number): boolean => {
    const recent = (seen.get(guestId) ?? []).filter(at => now - at < windowMs)

    if (recent.length >= limit) {
      seen.set(guestId, recent)

      return false
    }

    recent.push(now)
    seen.set(guestId, recent)

    return true
  }
}

export type PopupLimiter = ReturnType<typeof createPopupLimiter>

/** Decide what a guest's new-window request becomes: a preview tab request, or nothing. */
export function previewGuestNewWindowRequest(
  guestId: number,
  details: GuestWindowOpenDetails,
  allow: PopupLimiter,
  now: number
): { drop: string } | { request: PreviewGuestNewWindow } {
  if (!isPreviewTabUrl(details.url)) {
    return { drop: 'scheme' }
  }

  const post = postFrom(details)

  if (post === null) {
    return { drop: 'file-upload' }
  }

  if (!allow(guestId, now)) {
    return { drop: 'rate-limit' }
  }

  return {
    request: {
      background: details.disposition === 'background-tab',
      guestId,
      ...(post ? { post } : {}),
      url: details.url
    }
  }
}

interface GuestLike {
  id: number
  isDestroyed(): boolean
  setWindowOpenHandler(handler: (details: GuestWindowOpenDetails) => { action: 'deny' }): void
}

interface HostLike {
  isDestroyed(): boolean
  send(channel: string, payload: PreviewGuestNewWindow): void
}

/** Install the always-deny handler on one preview guest; forwards admitted requests to `host`. */
export function installPreviewGuestNewWindow(
  guest: GuestLike,
  host: HostLike,
  allow: PopupLimiter,
  log: (line: string) => void = () => {},
  now: () => number = Date.now
): void {
  guest.setWindowOpenHandler(details => {
    const outcome = previewGuestNewWindowRequest(guest.id, details, allow, now())

    if ('drop' in outcome) {
      log(`[preview] new-tab request dropped (${outcome.drop})`)
    } else if (!host.isDestroyed()) {
      host.send(PREVIEW_GUEST_NEW_WINDOW_CHANNEL, outcome.request)
    }

    return { action: 'deny' }
  })
}

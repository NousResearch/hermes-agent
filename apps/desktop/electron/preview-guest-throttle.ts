import type { Session, WebContents } from 'electron'

/** Partition the preview pane's `<webview>` always carries. */
export const PREVIEW_GUEST_PARTITION = 'persist:hermes-preview'

export interface PreviewGuestAttachPrefs {
  backgroundThrottling?: boolean
  preload?: string
}

/** Keep the preload partition-scoped; idle guests use Chromium's throttling. */
export function applyPreviewGuestAttachPrefs(
  webPreferences: PreviewGuestAttachPrefs,
  partition: string | undefined,
  preloadPath: string
): void {
  if (partition !== PREVIEW_GUEST_PARTITION) {
    return
  }

  webPreferences.preload = preloadPath
  webPreferences.backgroundThrottling = true
}

let nextLease = 0

const leases = new Map<
  number,
  {
    owner: WebContents
    guest: WebContents
    release: () => void
    cancelled: AbortController
  }
>()

const activeGuests = new WeakMap<WebContents, Set<number>>()

/**
 * Only the embedding window can temporarily wake its preview guest. Scripts
 * may await rAF (including user-approved plugin scripts), but keeping that
 * exemption for the guest's lifetime also keeps the containing window painting.
 * No deadline here: each caller owns its existing timeout/cancellation policy.
 */
export function beginPreviewGuestActivity(
  owner: WebContents,
  guest: undefined | WebContents,
  previewSession: Session,
  holdOwner: () => () => void
): number {
  if (
    owner.isDestroyed() ||
    owner.getType() !== 'window' ||
    !guest ||
    guest.isDestroyed() ||
    guest.getType() !== 'webview' ||
    guest.hostWebContents !== owner ||
    guest.session !== previewSession
  ) {
    throw new Error('Preview guest does not belong to this window')
  }

  let tokens = activeGuests.get(guest)

  if (!tokens) {
    tokens = new Set()
    activeGuests.set(guest, tokens)
  }

  const token = ++nextLease
  const group = tokens
  const original = guest.getBackgroundThrottling()

  // Only the first lease owns lifecycle listeners and restoration. Later
  // leases reuse its release function, so finishing one cannot suspend another.
  let release = group.size ? leases.get(group.values().next().value!)!.release : undefined

  if (!release) {
    // Guest frames alone do not wake the host's typing/scroll timers.
    // The shared window controller must arbitrate this with live chat work.
    const releaseOwner = holdOwner()

    const restore = () => {
      if (group.size) {
        return
      }

      activeGuests.delete(guest)
      guest.removeListener('destroyed', discard)
      guest.removeListener('render-process-gone', discard)
      owner.removeListener('destroyed', discard)
      owner.removeListener('render-process-gone', discard)
      owner.removeListener('did-navigate', discard)

      try {
        if (!guest.isDestroyed()) {
          guest.setBackgroundThrottling(original)
        }
      } finally {
        releaseOwner()
      }
    }

    const discard = () => {
      for (const id of group) {
        leases.get(id)?.cancelled.abort(new Error('Preview guest is unavailable'))
        leases.delete(id)
      }

      group.clear()
      restore()
    }

    release = restore
    guest.once('destroyed', discard)
    guest.once('render-process-gone', discard)
    owner.once('destroyed', discard)
    owner.once('render-process-gone', discard)
    owner.on('did-navigate', discard)

    try {
      guest.setBackgroundThrottling(false)
    } catch (error) {
      discard()
      throw error
    }
  }

  group.add(token)
  leases.set(token, { owner, guest, release, cancelled: new AbortController() })

  return token
}

export function endPreviewGuestActivity(owner: WebContents, token: number): void {
  const lease = leases.get(token)

  if (!lease || lease.owner !== owner || lease.guest.hostWebContents !== owner) {
    return
  }

  leases.delete(token)
  activeGuests.get(lease.guest)?.delete(token)
  lease.cancelled.abort(new Error('Preview activity ended'))
  lease.release()
}

/** Dispatch in the loaded document, not through executeJavaScript's hidden
 * navigation queue. Old-document promises need explicit cancellation: Electron
 * does not guarantee they settle when that document disappears. */
export async function executePreviewGuestScript(owner: WebContents, token: number, code: string): Promise<unknown> {
  const lease = leases.get(token)

  if (!lease || lease.owner !== owner || lease.guest.hostWebContents !== owner) {
    throw new Error('Preview activity does not belong to this window')
  }

  const { guest, cancelled } = lease
  const { signal } = cancelled
  let abort!: () => void
  let loaded: (() => void) | undefined
  let navigated: (() => void) | undefined

  const interrupted = new Promise<never>((_, reject) => {
    abort = () => reject(signal.reason)
    signal.addEventListener('abort', abort, { once: true })
  })

  try {
    signal.throwIfAborted()

    if (!guest.getURL() || guest.isLoadingMainFrame()) {
      const ready = new Promise<void>(resolve => {
        loaded = resolve
        guest.once('did-stop-loading', loaded)
      })

      await Promise.race([ready, interrupted])
    }

    signal.throwIfAborted()
    navigated = () => cancelled.abort(new Error('Preview page navigated'))
    guest.once('did-navigate', navigated)
    // Navigation can replace the hidden widget without waking it, even while
    // getBackgroundThrottling() still reports false. Reapply to this document.
    guest.setBackgroundThrottling(false)

    // Readiness and dispatch share this main-process turn. Keep the existing
    // structured-clone result contract; mainFrame.executeJavaScript converts
    // values such as Maps, Dates, typed arrays and undefined to JSON-like values.
    return await Promise.race([guest.executeJavaScript(code), interrupted])
  } finally {
    signal.removeEventListener('abort', abort)

    if (loaded) {
      guest.removeListener('did-stop-loading', loaded)
    }

    if (navigated) {
      guest.removeListener('did-navigate', navigated)
    }

    endPreviewGuestActivity(owner, token)
  }
}

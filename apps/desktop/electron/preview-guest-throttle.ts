import type { Session, WebContents } from 'electron'

/**
 * The launch/default preview jar. A tab owned by the default profile (or an
 * empty profile) rides this partition so an upgrade never signs the existing
 * single-jar user out. Every other profile rides a `${PREVIEW_GUEST_PARTITION}-<sanitized>`
 * sibling jar (`previewPartitionForProfile`), so one profile's real-profile
 * sign-ins can never be presented in — or cleared from — another profile's tab
 * (#150283 P1-3, mirrors oauth-partition.ts's per-connection jars).
 */
export const PREVIEW_GUEST_PARTITION = 'persist:hermes-preview'

// Colon-free ON PURPOSE (same invariant as oauth-partition.ts): Electron
// percent-escapes ':' to '%3A' in the on-disk profile folder, and a Windows
// folder whose name contains '%3A' gets a cookie store the network stack can
// neither read nor write. encodeURIComponent then '%'→'_' keeps the whole
// component within the characters Electron never escapes.
function sanitizePartitionComponent(id: string): string {
  return encodeURIComponent(id).replace(/%/g, '_')
}

// Mirror of src/store/profile.ts `normalizeProfileKey` — kept here because this
// module stays free of renderer/electron imports so it unit-tests in the
// electron vitest project. The equivalence is pinned by a unit test.
function normalizePreviewProfileKey(profile: string | null | undefined): string {
  return (profile ?? '').trim() || 'default'
}

/**
 * The preview partition a tab owned by `profile` rides. The default/empty
 * profile keeps the bare launch jar (`PREVIEW_GUEST_PARTITION`) so the existing
 * single-jar user is not signed out by the upgrade; every other profile gets
 * its own sibling jar. Lane B imports this for the `<webview>` partition
 * attribute, so both sides MUST produce byte-identical names for a profile.
 */
export function previewPartitionForProfile(profile: string | null | undefined): string {
  const key = normalizePreviewProfileKey(profile)

  return key === 'default' ? PREVIEW_GUEST_PARTITION : `${PREVIEW_GUEST_PARTITION}-${sanitizePartitionComponent(key)}`
}

/**
 * The FAMILY predicate: true for the launch jar and every per-profile sibling
 * jar, and NOTHING else. The GHSA will-attach gate and the throttle/ipc
 * plumbing admit exactly this family — the preload and admission rules are
 * identical per jar, so the advisory's scheme/popup protections still hold —
 * and never widen to an arbitrary partition.
 */
export function isPreviewPartition(partition: string | undefined): boolean {
  return (
    partition === PREVIEW_GUEST_PARTITION ||
    (typeof partition === 'string' && partition.startsWith(`${PREVIEW_GUEST_PARTITION}-`))
  )
}

export interface PreviewGuestAttachPrefs {
  backgroundThrottling?: boolean
  preload?: string
}

// Every preview-family session the attach gate has admitted. A webview's
// `webContents.session` is the memoized `session.fromPartition(partition)`, so
// recording that object here lets the ACTIVITY gate accept any sibling preview
// jar by identity — without reading a partition name off a Session (Electron
// exposes none) and without widening to arbitrary partitions.
const previewGuestSessions = new WeakSet<Session>()

/** True when `session` is a preview-family jar the attach gate admitted (#150283 F1).
 *  The guest's own session is authoritative: a named profile rides a sibling jar,
 *  not the default one, so the activity gate must accept it. */
export function isPreviewGuestSession(session: Session | undefined): boolean {
  return Boolean(session && previewGuestSessions.has(session))
}

/** Keep the preload partition-scoped; idle guests use Chromium's throttling. */
export function applyPreviewGuestAttachPrefs(
  webPreferences: PreviewGuestAttachPrefs,
  partition: string | undefined,
  preloadPath: string,
  resolveSession?: (partition: string) => Session
): void {
  if (!isPreviewPartition(partition)) {
    return
  }

  webPreferences.preload = preloadPath
  webPreferences.backgroundThrottling = true

  // Remember the resolved preview jar so the activity gate recognizes this
  // guest's own (possibly non-default) session later.
  if (resolveSession && partition) {
    previewGuestSessions.add(resolveSession(partition))
  }
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
  holdOwner: () => () => void
): number {
  if (
    owner.isDestroyed() ||
    owner.getType() !== 'window' ||
    !guest ||
    guest.isDestroyed() ||
    guest.getType() !== 'webview' ||
    guest.hostWebContents !== owner ||
    !isPreviewGuestSession(guest.session)
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

    // A new document can replace the hidden widget while the stored flag stays
    // false. Lease-only work (input/capture) must wake it without another script.
    const wake = () => guest.setBackgroundThrottling(false)

    const restore = () => {
      if (group.size) {
        return
      }

      activeGuests.delete(guest)
      guest.removeListener('destroyed', discard)
      guest.removeListener('render-process-gone', discard)
      guest.removeListener('did-navigate', wake)
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
    guest.on('did-navigate', wake)
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
    // Backstop for a lease acquired during loading: reapply once the document
    // is ready, even if its navigation committed before the lease was acquired.
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

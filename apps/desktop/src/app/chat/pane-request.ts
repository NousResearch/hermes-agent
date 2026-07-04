import { ensureProfileSocketOpen } from '@/store/gateway'
import { ensureGatewayProfile } from '@/store/profile'
import { $sessions } from '@/store/session'
import { $splitOpen } from '@/store/split'
import { isSecondaryWindow } from '@/store/windows'

import { MAIN_PANE_VIEW } from './pane-view'

// ── Profile-pinned gateway request wrappers (design §4) ─────────────────────
//
// Invariant P1: sends/resumes execute only from the ACTIVE pane, whose
// activation already swapped the active gateway to its profile. These wrappers
// close the remaining holes:
//
// 1. Enter-beats-swap — activation `void`s its ensureGatewayProfile, so a
//    keystroke-fast submit could race out over the previous pane's socket.
//    The active branch awaits the (serialized, no-op-when-settled) swap first.
// 2. Background-pane ops — a pane can issue RPCs while INACTIVE (the split's
//    boot restore; the MAIN pane's reconnect route-resume, bounded auto-retry
//    and queue auto-drain while the split is focused). Those must never ride
//    the active gateway — with the panes on different profiles it belongs to
//    the OTHER pane — so they go out over their own profile's pinned socket
//    (opened/reused without touching the active pointer).

export type PaneGatewayRequest = <T>(
  method: string,
  params?: Record<string, unknown>,
  timeoutMs?: number,
  signal?: AbortSignal
) => Promise<T>

interface PaneRequestDeps {
  /** The pane's session profile tag (null = default profile / untagged). */
  getProfile: () => string | null
  isActive: () => boolean
  /** The shared active-gateway request (use-gateway-request). */
  requestGateway: PaneGatewayRequest
}

/** The SPLIT pane's request fn: active → settle the profile swap, then the
 *  shared active-gateway path; inactive → the profile's own pinned socket. */
export function createPaneRequest({ getProfile, isActive, requestGateway }: PaneRequestDeps): PaneGatewayRequest {
  return async <T>(method: string, params: Record<string, unknown> = {}, timeoutMs?: number, signal?: AbortSignal) => {
    const profile = getProfile()

    if (isActive()) {
      // No-op fast path when already on-profile; a null tag still awaits any
      // in-flight swap (ensureGatewayProfile's "no explicit profile" branch),
      // which is exactly the race being closed.
      await ensureGatewayProfile(profile)

      return requestGateway<T>(method, params, timeoutMs, signal)
    }

    const socket = await ensureProfileSocketOpen(profile)

    if (!socket) {
      throw new Error('Hermes gateway unavailable')
    }

    return socket.request<T>(method, params, timeoutMs, signal)
  }
}

/**
 * The MAIN pane's session profile: its selected session's row tag. Null for a
 * fresh draft ("no explicit profile" → ensureGatewayProfile only settles any
 * in-flight swap and keeps the current gateway — today's semantics). Also
 * used by SplitChatPane's switch-back activation to restore the gateway.
 *
 * Reads the UNMIRRORED selection: while the split pane is focused the
 * $selectedStoredSessionId singleton mirrors the SPLIT's session
 * (split-mirror.ts), so the main pane's truth is the cache ref the controller
 * binds into MAIN_PANE_VIEW — written synchronously by every main-pane
 * resume/draft/create. Split closed, ref and atom are always identical.
 */
export function mainPaneSessionProfile(): string | null {
  const selected = MAIN_PANE_VIEW.selectedStoredSessionIdRef.current

  if (!selected) {
    return null
  }

  return (
    $sessions.get().find(session => session.id === selected || session._lineage_root_id === selected)?.profile ?? null
  )
}

/**
 * The MAIN pane's request fn. With the split closed this is a pure
 * pass-through — byte-identical single-pane behavior. While the split is open:
 *
 * - main pane ACTIVE → the same swap-settling guard as the split's wrapper,
 *   so a send fired the instant focus returns can't race the activation swap
 *   back onto the split's socket;
 * - main pane BACKGROUND (split focused) → the main session's own profile
 *   socket. The active gateway belongs to the split, and the non-interactive
 *   main-pane dispatchers (reconnect route-resume, bounded resume auto-retry,
 *   composer queue auto-drain) keep firing while the split is focused — with
 *   the panes on different profiles their RPCs would otherwise land on the
 *   split profile's backend and "session not found".
 */
export function createMainPaneRequest({ requestGateway }: { requestGateway: PaneGatewayRequest }): PaneGatewayRequest {
  return async <T>(method: string, params: Record<string, unknown> = {}, timeoutMs?: number, signal?: AbortSignal) => {
    // Secondary windows share localStorage, so their $splitOpen snapshot can
    // read true for a split only the main window renders — stay pass-through.
    if (!$splitOpen.get() || isSecondaryWindow()) {
      return requestGateway<T>(method, params, timeoutMs, signal)
    }

    const profile = mainPaneSessionProfile()

    if (MAIN_PANE_VIEW.isActive()) {
      await ensureGatewayProfile(profile)

      return requestGateway<T>(method, params, timeoutMs, signal)
    }

    // A profile-less main pane (fresh draft) has no pinned socket to prefer,
    // and nothing session-bound can be in flight for a draft — keep the
    // shared path rather than guessing a profile.
    if (profile === null) {
      return requestGateway<T>(method, params, timeoutMs, signal)
    }

    const socket = await ensureProfileSocketOpen(profile)

    if (!socket) {
      throw new Error('Hermes gateway unavailable')
    }

    return socket.request<T>(method, params, timeoutMs, signal)
  }
}

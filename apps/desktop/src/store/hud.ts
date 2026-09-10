/**
 * HUD mode — the chrome-free floating chat.
 *
 * A transparent, frameless, always-on-top window showing nothing but the REAL
 * composer with the reply scrolling above it, so Hermes can be driven while
 * the user works in another app (Figma, a browser).
 *
 * It is NOT a puppet window. Unlike the pet overlay / quick entry, the HUD is
 * a full app renderer with its own gateway — the same thing `openWindow()`
 * spawns, just reshaped. That is the whole design: the HUD renders `ChatView`
 * in its `hud` variant, so the composer IS the app's composer (attachments,
 * slash commands, queue, voice, model pill) rather than a lookalike that
 * drifts. This module owns only the mode flag and the window lifecycle.
 */

import { atom } from 'nanostores'

import { requestComposerDraftSync } from '@/store/composer'
import { $activeConnectionId } from '@/store/connections'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import { $sessions, knownSessionOwner, rememberedSessionProfile } from '@/store/session'
import { isHudWindow } from '@/store/windows'

/** Whether a HUD window is currently up. In the HUD's own renderer this is
 *  always true (it IS the HUD); in the main window it tracks the child so the
 *  titlebar toggle reads correctly.
 *
 *  Deliberately NOT persisted. The HUD is a live window main owns, so it can
 *  never outlive the app — a remembered `true` from the last run just makes the
 *  first toggle a no-op ("the button does nothing after a restart"). Main
 *  broadcasts the truth on every change, which is the only authority there is. */
export const $hudActive = atom(isHudWindow())

/** True only in the HUD window itself — the renderer flag that swaps the app
 *  shell for the slim floating layout. Constant for the window's life, so it
 *  never invalidates a render path mid-session. */
export const $hudMode = atom(isHudWindow())

/** Which conversation the HUD is showing, as far as this window knows. Lets the
 *  toggle tell "switch the HUD to this tab" apart from "dismiss the HUD". */
export const $hudSession = atom<null | string>(null)

/** True when the shell exposes HUD mode (desktop only). */
export const canUseHud = (): boolean =>
  typeof window !== 'undefined' && typeof window.hermesDesktop?.hud?.open === 'function'

export function openHud(sessionId?: null | string): void {
  const api = window.hermesDesktop?.hud

  if (!api) {
    return
  }

  // Push whatever is half-typed here into the shared draft stash BEFORE the
  // HUD window exists, so its composer boots with the text rather than racing
  // a cross-window storage event that lands after it has already painted.
  requestComposerDraftSync('flush')

  // Which backend the HUD must boot against. A connection-tagged session owns
  // an exact `(connectionId, profile)` route; use that before the active route.
  // Bare/unknown sessions retain the presentation fallback for legacy chats.
  const owner = knownSessionOwner($sessions.get(), sessionId ?? null)
  const ownerRoute = owner && typeof owner === 'object' ? owner : null

  const profile = normalizeProfileKey(
    ownerRoute?.profile ??
      (typeof owner === 'string'
        ? owner
        : rememberedSessionProfile($sessions.get(), sessionId ?? null, $activeGatewayProfile.get()))
  )

  const connectionId = ownerRoute?.connectionId?.trim() || $activeConnectionId.get()

  $hudActive.set(true)
  $hudSession.set(sessionId ?? null)
  void api.open({
    sessionId: sessionId ?? null,
    profile,
    ...(connectionId ? { connectionId } : {})
  })
}

/** Leave HUD mode. Callable from either window — main closes the child, the
 *  HUD closes itself; both restore the app window. */
export function closeHud(): void {
  const api = window.hermesDesktop?.hud

  if (!api) {
    return
  }

  $hudActive.set(false)
  $hudSession.set(null)
  void api.close()
}

/**
 * Enter HUD mode, leave it, or point the open HUD at another conversation.
 *
 * The retarget rung is why `$hudSession` exists: asking for HUD mode from a tab
 * the HUD is NOT on means "put this conversation in the HUD", and main already
 * implements that (openHudWindow sends `hud:goto`, or respawns across a profile
 * boundary). Reading `$hudActive` alone never reached it — the toggle dismissed
 * the window instead, so the retarget path was unreachable from the UI.
 *
 * Two cases still dismiss, deliberately:
 * - No target (a fresh draft with nothing selected). There is nothing to
 *   retarget onto, and "the toggle stopped closing the HUD" is the worse bug.
 * - Inside the HUD's own window, where the toggle is the way OUT. Its renderer
 *   never subscribes to the state broadcast (useHudHandoff returns early there),
 *   so `$hudSession` is always null and every target would read as a retarget —
 *   leaving the exit keybind unable to exit.
 */
export function toggleHud(sessionId?: null | string): void {
  if (!$hudActive.get()) {
    openHud(sessionId)

    return
  }

  const target = sessionId ?? null

  if (!target || isHudWindow() || target === $hudSession.get()) {
    closeHud()

    return
  }

  openHud(target)
}

/** Restore the HUD's persisted geometry to its display-aware default. */
export function resetHudLayout(): void {
  void window.hermesDesktop?.hud?.resetLayout?.()
}

/** Tell main which session this HUD is on. Main holds it (the HUD's renderer
 *  doesn't outlive the window) and hands it back in the close broadcast so the
 *  app window knows what to re-home onto. */
export const reportHudSession = (sessionId: null | string): void => window.hermesDesktop?.hud?.setSession?.(sessionId)

/**
 * Track the HUD window's real state so the titlebar toggle can't go stale when
 * the HUD is closed from its own side (⌘W, its exit button), and hand the
 * app window the session the HUD ended on. Returns a disposer; no-ops outside
 * Electron.
 */
export function watchHudState(onClosed?: (sessionId: null | string) => void): () => void {
  const off = window.hermesDesktop?.hud?.onChanged?.(({ open, sessionId }) => {
    $hudActive.set(open)
    $hudSession.set(open ? sessionId : null)

    if (!open) {
      onClosed?.(sessionId)
    }
  })

  return off ?? (() => {})
}

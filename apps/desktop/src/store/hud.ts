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

import { readJson, writeJson } from '@/lib/storage'
import { requestComposerDraftSync } from '@/store/composer'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import { $sessions, rememberedSessionProfile } from '@/store/session'
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

// ── Orientation ──────────────────────────────────────────────────────────────

/** Where the composer sits relative to the transcript band.
 *  - `auto`: edge-aware — the band grows AWAY from the screen edge the HUD is
 *    parked against (parked at the top the composer hugs the top and the
 *    transcript hangs below; anywhere else the composer sits at the bottom
 *    and the transcript grows upward). This was the layout until the
 *    always-below experiment pinned it.
 *  - `composer-top`: the Spotlight shape — composer pinned to the window's
 *    top edge, transcript always hanging below, wherever the HUD is parked.
 *  - `composer-bottom`: mirrored — composer pinned to the bottom edge,
 *    transcript always above it. */
export type HudOrientation = 'auto' | 'composer-top' | 'composer-bottom'

export const HUD_ORIENTATIONS: readonly HudOrientation[] = ['auto', 'composer-top', 'composer-bottom']

/** Device-local presentation preference — purely this window's layout, so it
 *  lives in localStorage (the renderer owns it), scoped under one key. */
const HUD_ORIENTATION_KEY = 'hermes.desktop.hud.orientation'

const isHudOrientation = (value: unknown): value is HudOrientation =>
  typeof value === 'string' && (HUD_ORIENTATIONS as readonly string[]).includes(value)

/** The user's orientation choice. The default is `composer-top` — the shipped
 *  Spotlight shape is preserved verbatim for everyone who never opens the
 *  setting; `auto` restores the original edge-aware layout and
 *  `composer-bottom` pins the bar to the bottom edge. */
export const $hudOrientation = atom<HudOrientation>(readPersistedHudOrientation())

function readPersistedHudOrientation(): HudOrientation {
  const stored = readJson<unknown>(HUD_ORIENTATION_KEY)

  return isHudOrientation(stored) ? stored : 'composer-top'
}

export function setHudOrientation(orientation: HudOrientation): void {
  $hudOrientation.set(orientation)
  writeJson(HUD_ORIENTATION_KEY, orientation)
}

export function openHud(sessionId?: null | string): void {
  const api = window.hermesDesktop?.hud

  if (!api) {
    return
  }

  // Push whatever is half-typed here into the shared draft stash BEFORE the
  // HUD window exists, so its composer boots with the text rather than racing
  // a cross-window storage event that lands after it has already painted.
  requestComposerDraftSync('flush')

  // Which backend the HUD must boot against. The HUD is a full renderer that
  // adopts the PRIMARY backend's profile by default, so handing it a session
  // from a non-primary profile without saying so resolves the id against the
  // wrong backend — the lookup misses and the HUD falls back to the default
  // profile's last session (#82285). Same ladder the remembered-navigation key
  // uses: the session's stamped owner wins, and a fresh/unstamped/uncached
  // target inherits the profile the user is looking at.
  const profile = normalizeProfileKey(
    rememberedSessionProfile($sessions.get(), sessionId ?? null, $activeGatewayProfile.get())
  )

  $hudActive.set(true)
  $hudSession.set(sessionId ?? null)
  void api.open({ sessionId: sessionId ?? null, profile })
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

export const toggleHud = (sessionId?: null | string) => ($hudActive.get() ? closeHud() : openHud(sessionId))

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

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

import type { HudAskPayload, HudLaunchOptions, HudPrefs, HudPrefsStatus, HudRoomFeed } from '@/lib/hud-prefs'
import { parseHudVoiceCommand } from '@/lib/hud-voice-command'
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

/** Start HUD mode on a fresh draft owned by an explicit agent profile. Unlike
 * openHud(null), this does not infer the target from the primary window's
 * active profile — the pointer launcher may choose a different agent while the
 * main workspace stays exactly where it is. */
export async function openHudForProfile(profile: string): Promise<boolean> {
  const api = window.hermesDesktop?.hud

  if (!api) {
    return false
  }

  requestComposerDraftSync('flush')
  $hudActive.set(true)
  $hudSession.set(null)

  try {
    const result = await api.open({ sessionId: null, profile: normalizeProfileKey(profile) })

    return result.ok === true
  } catch {
    $hudActive.set(false)

    return false
  }
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

// ── Follow / ask prefs, switchers ─────────────────────────────────────────────
//
// Main owns these (the OS chord, the optional input hook, the cursor poll);
// the renderer mirrors the status it broadcasts and asks for changes. Read in
// the HUD (toggle + pills), in Settings (rows), and in the app window's
// keybinds — one atom, one broadcast, no window disagrees.

export const $hudPrefs = atom<HudPrefsStatus | null>(null)

/** Agents + rooms for the HUD's switchers, as last pushed by the primary
 *  renderer for Quick Entry and cached in main. */
export const $hudLaunchOptions = atom<HudLaunchOptions>({ agents: [], groups: [] })

/** The ask sheet's payload while the sheet is up; null otherwise. */
export const $hudAsk = atom<HudAskPayload | null>(null)

export async function loadHudPrefs(): Promise<HudPrefsStatus | null> {
  const api = window.hermesDesktop?.hud

  if (!api?.getPrefs) {
    return null
  }

  try {
    const status = await api.getPrefs()
    $hudPrefs.set(status)

    return status
  } catch {
    return null
  }
}

/** Mirror main's status broadcasts. Returns a disposer; no-op outside Electron. */
export function watchHudPrefs(): () => void {
  return window.hermesDesktop?.hud?.onPrefs?.(status => $hudPrefs.set(status)) ?? (() => {})
}

export async function setHudPrefs(patch: Partial<HudPrefs>): Promise<HudPrefsStatus | null> {
  const api = window.hermesDesktop?.hud

  if (!api?.setPrefs) {
    return null
  }

  try {
    const status = await api.setPrefs(patch)
    $hudPrefs.set(status)

    return status
  } catch {
    return null
  }
}

export function toggleHudFollow(): void {
  void setHudPrefs({ follow: !($hudPrefs.get()?.follow ?? false) })
}

export async function refreshHudLaunchOptions(): Promise<HudLaunchOptions> {
  const api = window.hermesDesktop?.hud

  if (!api?.launchOptions) {
    return $hudLaunchOptions.get()
  }

  try {
    const options = await api.launchOptions()

    const next: HudLaunchOptions = {
      agents: Array.isArray(options?.agents) ? options.agents : [],
      groups: Array.isArray(options?.groups) ? options.groups : []
    }

    $hudLaunchOptions.set(next)

    return next
  } catch {
    return $hudLaunchOptions.get()
  }
}

/** Remote control: open a room in the app window. The HUD keeps its own
 *  session — rooms render in the pane tree the HUD does not have. */
export async function openHudRoom(groupId: string): Promise<boolean> {
  const api = window.hermesDesktop?.hud
  const id = groupId.trim()

  if (!api?.openRoom || !id) {
    return false
  }

  try {
    return (await api.openRoom(id)).ok === true
  } catch {
    return false
  }
}

/**
 * Step the HUD to the next/previous agent in launcher order, wrapping — the
 * HUD's answer to ⌘⇧] / ⌘⇧[. Goes through `openHudForProfile`, so the app
 * window's workspace is untouched and the HUD respawns against the chosen
 * profile's backend. Returns false when there was nowhere to step.
 */
export function hudCycleAgent(direction: -1 | 1): boolean {
  const agents = $hudLaunchOptions.get().agents.filter(agent => agent.reachable)

  if (agents.length < 2) {
    return false
  }

  const current = normalizeProfileKey($activeGatewayProfile.get())
  const index = agents.findIndex(agent => normalizeProfileKey(agent.profile) === current)
  const start = index < 0 ? (direction === 1 ? -1 : 0) : index
  const next = agents[(start + direction + agents.length) % agents.length]

  if (!next || normalizeProfileKey(next.profile) === current) {
    return false
  }

  void openHudForProfile(next.profile)

  return true
}

/** HUD side: receive ask payloads — the one parked before this renderer
 *  existed, then every live push. Returns a disposer. */
export function watchHudAsk(): () => void {
  const api = window.hermesDesktop?.hud

  if (!api) {
    return () => {}
  }

  void api
    .takePendingAsk?.()
    .then(payload => {
      if (payload) {
        $hudAsk.set(payload)
      }
    })
    .catch(() => undefined)

  return api.onAsk?.(payload => $hudAsk.set(payload)) ?? (() => {})
}

export const dismissHudAsk = (): void => $hudAsk.set(null)

/**
 * A whole-utterance HUD command ("HUD top left", "HUD follow me", "HUD come
 * here", "HUD hide"). Returns true when the text WAS such a command and has
 * been handed to main — the caller must then not submit it as a turn. The
 * HUD need not be open for "follow me" (a pref) but must be for the moves;
 * main answers ok:false then and the words fall through to the agent.
 */
export function runHudVoiceCommand(text: string): boolean {
  const command = parseHudVoiceCommand(text)
  const api = window.hermesDesktop?.hud

  if (!command || !api?.command) {
    return false
  }

  void api.command(command).catch(() => undefined)

  return true
}

// ── Room mode ─────────────────────────────────────────────────────────────────
//
// The HUD talking into a room. The room engine runs in the app window; the
// HUD posts lines through main and shows the feed main pushes back. While a
// room is entered the HUD's composer submits here instead of to the agent
// session (see use-prompt-actions/submit).

/** The room the HUD is talking into, or null for the ordinary agent session. */
export const $hudRoom = atom<null | string>(null)

export const $hudRoomFeed = atom<HudRoomFeed | null>(null)

export async function enterHudRoom(groupId: string): Promise<boolean> {
  const api = window.hermesDesktop?.hud
  const id = groupId.trim()

  if (!api?.roomFeed || !id) {
    return false
  }

  $hudRoom.set(id)
  $hudRoomFeed.set(null)

  try {
    await api.watchRoom?.(id)
    const feed = await api.roomFeed(id)

    if ($hudRoom.get() === id) {
      $hudRoomFeed.set(feed && feed.groupId === id ? feed : null)
    }

    return true
  } catch {
    return $hudRoom.get() === id
  }
}

export function leaveHudRoom(): void {
  $hudRoom.set(null)
  $hudRoomFeed.set(null)
  void window.hermesDesktop?.hud?.watchRoom?.(null)?.catch(() => undefined)
}

/** Post a line into the entered room. False when not in a room, the text is
 *  blank, or the app window could not deliver it. */
export async function postHudRoom(text: string): Promise<boolean> {
  const api = window.hermesDesktop?.hud
  const room = $hudRoom.get()
  const trimmed = text.trim()

  if (!api?.roomPost || !room || !trimmed) {
    return false
  }

  try {
    return (await api.roomPost(room, trimmed)).ok === true
  } catch {
    return false
  }
}

/** HUD side: take feed pushes for the entered room. Returns a disposer. */
export function watchHudRoomFeed(): () => void {
  return (
    window.hermesDesktop?.hud?.onRoomFeed?.(feed => {
      if (feed && feed.groupId === $hudRoom.get()) {
        $hudRoomFeed.set(feed)
      }
    }) ?? (() => {})
  )
}

import { atom, computed } from 'nanostores'

import { readKey, writeKey } from '@/lib/storage'
import { $paneStates, setPaneOpen } from '@/store/panes'
import { $selectedStoredSessionId, $sessions } from '@/store/session'

// ── Split-pane (two-pane chat) state ────────────────────────────────────────
// The split pane is a SECOND chat view inside the main window. Which pane the
// user last interacted with is runtime-only ($activePaneId); which session the
// split holds is persisted so a relaunch restores it without a list scan. The
// split's open flag + width ride the existing store/panes.ts snapshot under
// SPLIT_PANE_ID, so resize/persistence come for free.

export type PaneId = 'main' | 'split'

/** store/panes.ts snapshot id for the split pane (open state + width). */
export const SPLIT_PANE_ID = 'chat-split'

const SPLIT_SESSION_KEY = 'hermes.desktop.splitPane.v1'

export interface SplitPaneSession {
  storedId: string
  /** Profile tag captured at open time so boot restore can route the probe
   *  without scanning every profile's session list. */
  profile: string | null
}

/** The pane the user last interacted with. Global surfaces (statusbar, rail,
 *  model menus) mirror this pane's session. Runtime-only — a relaunch always
 *  starts focused on the main pane. */
export const $activePaneId = atom<PaneId>('main')

function loadSplitSession(): SplitPaneSession | null {
  try {
    const raw = readKey(SPLIT_SESSION_KEY)

    if (!raw) {
      return null
    }

    const parsed = JSON.parse(raw) as { storedId?: unknown; profile?: unknown }

    if (typeof parsed.storedId !== 'string' || !parsed.storedId.trim()) {
      return null
    }

    return {
      storedId: parsed.storedId,
      profile: typeof parsed.profile === 'string' && parsed.profile.trim() ? parsed.profile : null
    }
  } catch {
    return null
  }
}

/** Which stored session the split pane holds, or null (closed / fresh draft). */
export const $splitPaneSession = atom<SplitPaneSession | null>(loadSplitSession())

$splitPaneSession.subscribe(value => {
  writeKey(SPLIT_SESSION_KEY, value ? JSON.stringify(value) : null)
})

// A split showing a brand-new draft (no stored session yet). Runtime-only: an
// unsent draft has nothing to restore, so a relaunch drops it.
export const $splitFreshDraft = atom(false)

/** Runtime id of the session the split pane currently VIEWS (maintained by the
 *  mounted SplitChatPane, null while the split is closed). Lets non-React
 *  modules — native notifications foremost — treat a split-visible session as
 *  on-screen without reaching into the pane's view bundle. Runtime-only. */
export const $splitPaneRuntimeSessionId = atom<string | null>(null)

// The MAIN pane's viewed runtime id, UNMIRRORED. While the split is focused
// the $activeSessionId singleton mirrors the SPLIT's session, so non-React
// consumers (native notifications) that must classify the main pane's
// on-screen transcript as foreground need the cache-ref truth. The controller
// registers a getter over its activeSessionIdRef (same pattern as the pane
// dispatcher below); null before it mounts, which reduces every consumer to
// the pre-split active-only logic.
let mainPaneRuntimeSessionIdGetter: (() => string | null) | null = null

export function registerMainPaneRuntimeSessionIdGetter(getter: (() => string | null) | null): void {
  mainPaneRuntimeSessionIdGetter = getter
}

export function mainPaneRuntimeSessionId(): string | null {
  return mainPaneRuntimeSessionIdGetter?.() ?? null
}

// Open = the pane snapshot says open AND there is something to show (a session
// or a fresh draft). Guards against a stale open flag surviving a session
// delete/restore-miss and mounting an empty split.
export const $splitOpen = computed(
  [$paneStates, $splitPaneSession, $splitFreshDraft],
  (paneStates, session, freshDraft) => Boolean(paneStates[SPLIT_PANE_ID]?.open) && (session !== null || freshDraft)
)

export function setActivePane(id: PaneId): void {
  if ($activePaneId.get() !== id) {
    $activePaneId.set(id)
  }
}

function sessionProfileTag(storedId: string): null | string {
  return (
    $sessions.get().find(session => session.id === storedId || session._lineage_root_id === storedId)?.profile ?? null
  )
}

/** Point the split pane at a stored session (pane-local "navigation").
 *
 * When the caller carries no profile, the tag is resolved rather than dropped:
 * same session → keep the current tag; different session → the $sessions row's
 * tag. Pane-local navigations that CREATE a session (fresh-draft send, branch
 * in the split) upsert the optimistic row — stamped with the live gateway
 * profile — before navigating, so the lookup lands even before a refresh. A
 * null tag would silently unpin the pane's request wrapper and drop the
 * profile from the gateway keep-set. */
export function setSplitPaneSession(storedId: string, profile?: string | null): void {
  const current = $splitPaneSession.get()

  $splitFreshDraft.set(false)
  $splitPaneSession.set({
    storedId,
    profile:
      profile !== undefined
        ? profile
        : current?.storedId === storedId
          ? current.profile
          : sessionProfileTag(storedId)
  })
}

/** Swap the split pane to a fresh new-chat draft (no stored session). */
export function setSplitPaneFreshDraft(): void {
  $splitPaneSession.set(null)
  $splitFreshDraft.set(true)
}

/**
 * Open the split pane. With a session → show that session; without → a fresh
 * draft. Enforces the "same session may not be open in both panes" invariant:
 * asking the split to show the main pane's current session just focuses the
 * main pane (the dispatcher applies the mirror rule for the other direction).
 */
export function openSplitPane(session?: SplitPaneSession | null): boolean {
  if (session) {
    if (session.storedId === $selectedStoredSessionId.get()) {
      setActivePane('main')

      return false
    }

    setSplitPaneSession(session.storedId, session.profile)
  } else if (!$splitPaneSession.get()) {
    setSplitPaneFreshDraft()
  }

  setPaneOpen(SPLIT_PANE_ID, true)
  setActivePane('split')

  return true
}

/** Close the split pane and hand focus back to the main pane. */
export function closeSplitPane(): void {
  setPaneOpen(SPLIT_PANE_ID, false)
  $splitPaneSession.set(null)
  $splitFreshDraft.set(false)
  setActivePane('main')
}

export function toggleSplitPane(): void {
  if ($splitOpen.get()) {
    closeSplitPane()
  } else {
    openSplitPane()
  }
}

/**
 * Point the split pane at a stored session, resolving the profile tag from the
 * loaded session list when the caller doesn't carry one — so the pane's
 * profile-pinned request wrapper and the gateway keep-set route correctly.
 * Raw open: the same-session invariant is enforced by openSplitPane against
 * the identity singletons; the controller's dispatcher (which also knows the
 * unmirrored main-pane selection) wraps this with its own guard.
 */
export function openSplitPaneForSession(storedId: string, profile?: string | null): boolean {
  const resolved = profile !== undefined ? profile : sessionProfileTag(storedId)

  return openSplitPane({ storedId, profile: resolved ?? null })
}

// ── Pane-open dispatcher registration ───────────────────────────────────────
// Session-open entry points that live outside DesktopController (command
// palette, keybind slots/switcher, session menus) can't reach its dispatcher
// through props. The controller registers it here; callers fall back to their
// legacy paths when no controller is mounted (early boot frames only).

interface PaneOpenDispatcher {
  /** §3.5 dispatcher: open in the focused pane; duplicates focus their pane. */
  openInActivePane: (storedId: string) => void
  /** "Open in split" with the controller's unmirrored same-session guard. */
  openInSplit: (storedId: string, profile?: string | null) => void
}

let paneOpenDispatcher: PaneOpenDispatcher | null = null

export function registerPaneOpenDispatcher(dispatcher: PaneOpenDispatcher | null): void {
  paneOpenDispatcher = dispatcher
}

/** Route a session open through the active-pane dispatcher. Returns false when
 *  none is registered — the caller then uses its legacy navigate() path. */
export function dispatchOpenSession(storedId: string): boolean {
  if (!paneOpenDispatcher) {
    return false
  }

  paneOpenDispatcher.openInActivePane(storedId)

  return true
}

/** "Open in split" entry point (session menus). Prefers the controller's
 *  dispatcher — it guards the same-session invariant against the UNMIRRORED
 *  main-pane selection — and falls back to the raw open before it mounts. */
export function openSessionInSplitPane(storedId: string, profile?: string | null): boolean {
  if (paneOpenDispatcher) {
    paneOpenDispatcher.openInSplit(storedId, profile)

    return true
  }

  return openSplitPaneForSession(storedId, profile)
}

// ── Split-pane model-selection delegate ─────────────────────────────────────
// The global ModelPickerOverlay lives outside any PaneViewContext, but its
// pick must land on the ACTIVE pane's model chip. The mounted SplitChatPane
// registers its pane-scoped selectModel here; the controller routes overlay
// picks through it while the split pane is focused. Null while the split is
// closed — overlay picks then take the main path, exactly as pre-split.

export interface PaneModelSelection {
  model: string
  provider: string
}

let splitPaneModelDelegate: ((selection: PaneModelSelection) => Promise<boolean>) | null = null

export function registerSplitPaneModelDelegate(
  delegate: ((selection: PaneModelSelection) => Promise<boolean>) | null
): void {
  splitPaneModelDelegate = delegate
}

export function splitPaneModelSelect(): ((selection: PaneModelSelection) => Promise<boolean>) | null {
  return splitPaneModelDelegate
}

/**
 * Boot-restore guard: keep the persisted split session only if it still
 * resolves. `resolve` is the caller's session probe (resolveStoredSession) —
 * injected so the store stays free of REST plumbing. A miss (null / throw)
 * closes the split instead of mounting a pane that can never resume.
 * Returns the validated session, or null when the split was (or is now) closed.
 */
export async function restoreSplitPaneSession(
  resolve: (storedId: string) => Promise<unknown>
): Promise<SplitPaneSession | null> {
  const session = $splitPaneSession.get()

  if (!session) {
    return null
  }

  try {
    const stored = await resolve(session.storedId)

    if (!stored) {
      closeSplitPane()

      return null
    }
  } catch {
    closeSplitPane()

    return null
  }

  return $splitPaneSession.get()
}

import { $activeSessionId, $selectedStoredSessionId } from '@/store/session'

import { MAIN_PANE_VIEW, type PaneSessionView } from './pane-view'

// ── Active-pane → global-singleton mirror ───────────────────────────────────
//
// Global surfaces scope themselves on the ACTIVE session's identity: blocking
// prompts (store/prompts.ts `$active` computeds, store/clarify.ts), native
// notification session-matching, preview attribution, the sidebar highlight,
// oneshot/compaction/background-delegation. Single-pane, the identity atoms
// and the main chat view are the same thing; with the split pane FOCUSED they
// must diverge — the globals follow the split's session while the main pane
// keeps rendering its own.
//
// The mirror therefore copies exactly the two IDENTITY atoms — nothing else:
//
// - `$activeSessionId` / `$selectedStoredSessionId` have no INTERACTIVE
//   concurrent writer while the split is focused: every user-driven main-pane
//   writer (resume, fresh draft, delete/archive re-route) is either
//   interaction-gated behind main-pane activation (pointerdown capture tears
//   this mirror down first) or wrapped by the controller to hand focus back to
//   the main pane before it runs. Background main-pane writers DO exist —
//   the reconnect route-resume and its bounded auto-retry rewrite the ids for
//   the main session while the split stays focused — which is why teardown
//   restores from the main pane's cache refs (always the live unmirrored
//   truth; every main-pane id write pairs the atom with its ref) instead of
//   an install-time snapshot that a background resume would invalidate.
// - The remaining view atoms ($messages, $busy, model family, cwd, clocks…)
//   keep live main-session writers (the state cache's main projection and
//   gateway-event.ts heartbeats), so mirroring them would interleave two
//   sessions' values into one atom. They stay main-pane truth; the split pane
//   reads its own bundle. Documented v1 line: identity follows focus, the
//   heavyweight workspace/status surfaces stay anchored to the main pane.
//
// Writes use raw `atom.set` — same effect as the store setters for these two
// atoms (no persistence side effects), and deliberately not routed through the
// MAIN pane setter block so a future persisting setter can't be reached from a
// background flush (design §3.6 / risk 6).

/**
 * Install the split-pane → globals identity mirror. Fires immediately (the
 * activation copy) and tracks every subsequent pane write. Teardown restores
 * the main pane's CURRENT identity from its cache refs (bound into
 * MAIN_PANE_VIEW by the controller), so a background main-pane resume that
 * rebound the runtime id mid-mirror is honored on switch-back.
 * Returns the teardown.
 */
export function mirrorPaneToGlobals(view: PaneSessionView): () => void {
  const unsubscribers = [
    view.$activeSessionId.subscribe(value => $activeSessionId.set(value)),
    view.$selectedStoredSessionId.subscribe(value => $selectedStoredSessionId.set(value))
  ]

  return () => {
    for (const unsubscribe of unsubscribers) {
      unsubscribe()
    }

    // Late ref lookup: bindMainPaneView may have swapped the ref objects in
    // after this mirror was created.
    $activeSessionId.set(MAIN_PANE_VIEW.activeSessionIdRef.current)
    $selectedStoredSessionId.set(MAIN_PANE_VIEW.selectedStoredSessionIdRef.current)
  }
}

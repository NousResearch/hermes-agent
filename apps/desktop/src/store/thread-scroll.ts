import { atom, type WritableAtom } from 'nanostores'

import { readKey, writeKey } from '@/lib/storage'
import { $activeProfile, normalizeProfileKey } from '@/store/profile'

// "Is the thread parked at the bottom" is owned by use-stick-to-bottom inside
// ThreadMessageList (the scroll container). That state lives only in that
// subtree, so ThreadMessageList mirrors it into these atoms for the composer,
// status stack, and floating jump button — all of which render OUTSIDE the thread.
//
// `$threadScrolledUp` dims the composer / status stack; `$threadJumpButtonVisible`
// shows the floating jump control. Both track `!isAtBottom` today, but stay
// separate so their thresholds can diverge again without touching consumers.
export const $threadScrolledUp = atom(false)
export const $threadJumpButtonVisible = atom(false)

// Skip no-op writes so subscribers don't churn on every scroll tick.
const setter = (target: WritableAtom<boolean>) => (value: boolean) => {
  if (target.get() !== value) {
    target.set(value)
  }
}

const setScrolledUp = setter($threadScrolledUp)
const setJumpButtonVisible = setter($threadJumpButtonVisible)

export const setThreadAtBottom = (isAtBottom: boolean) => {
  setScrolledUp(!isAtBottom)
  setJumpButtonVisible(!isAtBottom)
}

export const resetThreadScroll = () => setThreadAtBottom(true)

// Cross-component bridge: the jump button lives by the composer, the viewport's
// `scrollToBottom` lives inside the thread. The bridge registers a handler; the
// button fires it. Mirrors the composer focus/insert emitter pattern.
const handlers = new Set<() => void>()

export const onScrollToBottomRequest = (handler: () => void) => {
  handlers.add(handler)

  return () => void handlers.delete(handler)
}

export const requestScrollToBottom = () => handlers.forEach(handler => handler())

// Inline edit grows a sticky human bubble. Fire on pointerdown so the viewport
// escapes stick-to-bottom before focus/layout; close clears the edit flag when
// the inline composer unmounts.
const editOpenHandlers = new Set<() => void>()
const editCloseHandlers = new Set<() => void>()

export const onThreadEditOpen = (handler: () => void) => {
  editOpenHandlers.add(handler)

  return () => void editOpenHandlers.delete(handler)
}

export const notifyThreadEditOpen = () => editOpenHandlers.forEach(handler => handler())

export const onThreadEditClose = (handler: () => void) => {
  editCloseHandlers.add(handler)

  return () => void editCloseHandlers.delete(handler)
}

export const notifyThreadEditClose = () => editCloseHandlers.forEach(handler => handler())

// ── Per-session scroll position persistence ──────────────────────────────────
// When the user scrolls up to read history, their distance-from-bottom is
// saved keyed by sessionKey and profile. On return, the session-switch settle
// loop restores it instead of pinning to the bottom, so the reading position
// survives session switches. Offsets are stored as distance-from-bottom, not
// scrollTop: the render-budget backfill prepends older turns and the switch
// relayout reshapes content above the on-screen rows, and bottom-anchored math
// keeps the restored view steady under that churn — the same reason the
// "Show earlier" flow in list.tsx restores from the bottom edge.
export type ThreadScrollState = { kind: 'bottom' } | { fromBottom: number; kind: 'offset' }

export const THREAD_SCROLL_BOTTOM: ThreadScrollState = { kind: 'bottom' }

// Within this many pixels of the bottom edge counts as "parked at the bottom".
// Deliberately tight: use-stick-to-bottom's own near-bottom band re-locks lazy
// scrollers anyway, and recording a small real offset as `bottom` would yank a
// reader who stopped just shy of the edge.
export const THREAD_SCROLL_STICKY_THRESHOLD_PX = 8

export type ThreadScrollMetrics = {
  clientHeight: number
  scrollHeight: number
  scrollTop: number
}

export function threadScrollDistanceFromBottom(metrics: ThreadScrollMetrics): number {
  return Math.max(0, metrics.scrollHeight - metrics.scrollTop - metrics.clientHeight)
}

/** Classify live metrics as sticky-bottom or an exact reading offset. */
export function threadScrollStateFromMetrics(
  metrics: ThreadScrollMetrics,
  threshold = THREAD_SCROLL_STICKY_THRESHOLD_PX
): ThreadScrollState {
  const fromBottom = threadScrollDistanceFromBottom(metrics)

  return fromBottom <= threshold ? THREAD_SCROLL_BOTTOM : { fromBottom, kind: 'offset' }
}

/** The scrollTop that re-applies `state` at the current content height. */
export function threadScrollTargetTop(
  state: ThreadScrollState,
  metrics: Pick<ThreadScrollMetrics, 'clientHeight' | 'scrollHeight'>
): number {
  const max = Math.max(0, metrics.scrollHeight - metrics.clientHeight)

  return state.kind === 'bottom' ? max : Math.max(0, max - state.fromBottom)
}

// Storage is scoped per profile with the same `.profile.<encoded>` suffix the
// app's other persisted session state uses (session.ts profileNavigationKey),
// so two profiles can never read or evict each other's reading positions.
const SCROLL_POS_KEY_BASE = 'hermes.desktop.threadScroll.v1'

export function threadScrollStorageKey(profile: string): string {
  return `${SCROLL_POS_KEY_BASE}.profile.${encodeURIComponent(normalizeProfileKey(profile))}`
}

// Bounded so a marathon runtime that touches hundreds of sessions doesn't grow
// the map forever. JS object insertion order gives LRU eviction — saving
// delete-and-re-adds the key, so the front is always the least-recently-used.
export const THREAD_SCROLL_MEMORY_LIMIT = 120

function isValidState(value: unknown): value is ThreadScrollState {
  if (!value || typeof value !== 'object') {
    return false
  }

  const record = value as Record<string, unknown>

  if (record.kind === 'bottom') {
    return true
  }

  return record.kind === 'offset' && typeof record.fromBottom === 'number' && Number.isFinite(record.fromBottom)
}

function loadPositions(profile: string): Record<string, ThreadScrollState> {
  const raw = readKey(threadScrollStorageKey(profile))

  if (!raw) {
    return {}
  }

  try {
    const parsed = JSON.parse(raw) as unknown

    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      return {}
    }

    return Object.fromEntries(
      Object.entries(parsed as Record<string, unknown>).filter(
        (entry): entry is [string, ThreadScrollState] => isValidState(entry[1])
      )
    )
  } catch {
    return {}
  }
}

function persistPositions(profile: string, positions: Record<string, ThreadScrollState>) {
  const keys = Object.keys(positions)

  while (keys.length > THREAD_SCROLL_MEMORY_LIMIT) {
    delete positions[keys[0]!]
    keys.shift()
  }

  writeKey(threadScrollStorageKey(profile), keys.length === 0 ? null : JSON.stringify(positions))
}

export function getThreadScrollPosition(sessionKey: string): ThreadScrollState | undefined {
  return loadPositions($activeProfile.get())[sessionKey]
}

export function saveThreadScrollPosition(sessionKey: string, state: ThreadScrollState) {
  const profile = $activeProfile.get()
  const positions = loadPositions(profile)

  // Delete then re-add to track recency (insertion order = LRU anchor).
  delete positions[sessionKey]
  positions[sessionKey] = state
  persistPositions(profile, positions)
}

export function clearThreadScrollPosition(sessionKey: string) {
  const profile = $activeProfile.get()
  const positions = loadPositions(profile)

  if (positions[sessionKey] === undefined) {
    return
  }

  delete positions[sessionKey]
  persistPositions(profile, positions)
}

/**
 * The restore/record gate for the session-switch settle loop. Pure so the
 * warm/cold switch lifecycle is testable without a DOM:
 *
 * - cold (no transcript yet): forget any in-flight restore, do not record —
 *   an empty-transcript instance holds the PREVIOUS session's live state and
 *   must not file it under the new key.
 * - same key, already settled: the restore is done; keep recording only.
 * - same key, still settling: a dep identity change re-ran the effect
 *   mid-loop — re-arm the restore instead of stranding the viewport.
 * - anything else (first content for this key, or a key change): restore.
 */
export type ThreadScrollRestorePlan = { cold: boolean; gate: string | null | undefined; restore: boolean }

export function planThreadScrollRestore(
  prevGate: string | null | undefined,
  sessionKey: string | null | undefined,
  hasGroups: boolean,
  settled: boolean
): ThreadScrollRestorePlan {
  if (!hasGroups) {
    return { cold: true, gate: null, restore: false }
  }

  if (prevGate === sessionKey && settled) {
    return { cold: false, gate: sessionKey, restore: false }
  }

  return { cold: false, gate: sessionKey, restore: true }
}

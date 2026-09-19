import { atom, type WritableAtom } from 'nanostores'

import { activeConnectionScopeSuffix } from '@/lib/connection-scoped'
import { readKey, writeKey } from '@/lib/storage'
import { $activeProfile, normalizeProfileKey } from '@/store/profile'

// "Is the thread parked at the bottom" is owned by use-stick-to-bottom inside
// ThreadMessageList (the scroll container). That state lives only in that
// subtree, so ThreadMessageList mirrors it into these atoms for the composer,
// status stack, and floating jump button — all of which render OUTSIDE the thread.
//
// Each session owns its live chrome, including when two split panes are visible.
// Hidden keep-alive panes may neither publish nor reset a visible pane's state.
// Keep the flags separate so their thresholds can diverge without changing consumers.
export const $threadScrolledUpBySession = atom<Record<string, boolean>>({})
export const $threadJumpButtonVisibleBySession = atom<Record<string, boolean>>({})
export const $threadMessagesBelowBySession = atom<Record<string, number>>({})

export const publishThreadMessagesBelow = (
  count: number,
  publisher: { paneVisible: boolean; sessionId?: string | null }
): void => {
  if (publisher.paneVisible) {
    setSessionValue($threadMessagesBelowBySession, publisher.sessionId, count, 0)
  }
}

// Skip no-op writes and remove default entries so scroll ticks don't churn subscribers.
// Missing identities are not a shared bucket: callers use their existing surface id
// until a runtime session exists.
function setSessionValue<T extends boolean | number>(
  target: WritableAtom<Record<string, T>>,
  sessionId: string | null | undefined,
  value: T,
  empty: T
): void {
  if (!sessionId) {
    return
  }

  const current = target.get()

  if ((current[sessionId] ?? empty) === value) {
    return
  }

  const next = { ...current }

  if (value === empty) {
    delete next[sessionId]
  } else {
    next[sessionId] = value
  }

  target.set(next)
}

export const setThreadAtBottom = (isAtBottom: boolean, sessionId: string | null = null) => {
  setSessionValue($threadScrolledUpBySession, sessionId, !isAtBottom, false)
  setSessionValue($threadJumpButtonVisibleBySession, sessionId, !isAtBottom, false)
}

export const resetThreadScroll = (sessionId: string | null = null) => {
  setThreadAtBottom(true, sessionId)
  setSessionValue($threadMessagesBelowBySession, sessionId, 0, 0)
}

export const publishThreadAtBottom = (
  isAtBottom: boolean,
  publisher: { paneVisible: boolean; sessionId?: string | null }
): void => {
  if (!publisher.paneVisible) {
    return
  }

  setThreadAtBottom(isAtBottom, publisher.sessionId)
}

export const resetPublishedThreadScroll = (publisher: { paneVisible: boolean; sessionId?: string | null }): void => {
  if (!publisher.paneVisible) {
    return
  }

  resetThreadScroll(publisher.sessionId)
}

// Cross-component bridge: the jump button lives by the composer, the viewport's
// `scrollToBottom` lives inside the thread. The bridge registers a handler; the
// button fires it. Mirrors the composer focus/insert emitter pattern.
const handlers = new Map<string | null, Set<() => void>>()

export const onScrollToBottomRequest = (handler: () => void, sessionId: string | null = null) => {
  const scoped = handlers.get(sessionId) ?? new Set<() => void>()

  scoped.add(handler)
  handlers.set(sessionId, scoped)

  return () => {
    scoped.delete(handler)

    if (scoped.size === 0) {
      handlers.delete(sessionId)
    }
  }
}

export const requestScrollToBottom = (sessionId: string | null = null) => {
  handlers.get(sessionId)?.forEach(handler => handler())
}

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

// Composer metrics write --composer-measured-height onto the chat surface,
// which grows [data-slot="aui_composer-clearance"] and can shrink the
// clampToComposer viewport. The post-settle restore ResizeObserver sees that
// as a content resize and used to re-pin a frozen fromBottom — rewriting
// scrollTop on every keystroke. Transcript height is scrollHeight minus the
// clearance spacer, so composer-only layout is distinguishable from real
// message/prepend/streaming growth.
export type ThreadScrollRestoreResizeMetrics = {
  clearanceHeight: number
  clientHeight: number
  scrollHeight: number
}

export function threadScrollTranscriptHeight(
  metrics: Pick<ThreadScrollRestoreResizeMetrics, 'clearanceHeight' | 'scrollHeight'>
): number {
  return Math.max(0, metrics.scrollHeight - Math.max(0, metrics.clearanceHeight))
}

/**
 * Post-settle restore RO may re-pin the restored target only when transcript
 * rows actually changed height. Composer clearance / viewport-box resizes and
 * no-op RO deliveries must not rewrite scrollTop — for a bottom target either,
 * or every keystroke yanks a view the user moved off the bottom back down.
 */
export function shouldReapplyFrozenThreadScrollOffset(
  target: ThreadScrollState,
  settled: boolean,
  previous: Pick<ThreadScrollRestoreResizeMetrics, 'clearanceHeight' | 'scrollHeight'>,
  next: Pick<ThreadScrollRestoreResizeMetrics, 'clearanceHeight' | 'scrollHeight'>
): boolean {
  if (!settled) {
    return false
  }

  const before = Math.round(threadScrollTranscriptHeight(previous))
  const after = Math.round(threadScrollTranscriptHeight(next))

  // A bottom target follows growth only — a shrink is clamped by the browser.
  return target.kind === 'bottom' ? after > before : after !== before
}

// ── Turn-end viewport anchor (#108941) ───────────────────────────────────────
// Where the transcript lands when a turn ENDS. The stick-to-bottom lock parks
// the view at the very end, so on a long turn — heavy tool activity, a long
// answer — the answer's opening lines sit off-screen and the reader scrolls UP
// to find where the answer begins. 'prompt' settles at the newest prompt's own
// position instead, so the exchange reads top-down from the question; 'bottom'
// is the landing every existing install already has.
//
// Presentation-only, so the renderer owns it (desktop AGENTS.md: state lives
// with its authority) — no backend config key, no gateway mirror. The reader
// always wins: the anchor only applies while the viewport is still parked at
// the bottom, so scrolling up mid-turn keeps the reading position in either
// mode (and the existing "N messages below" jump pill stays the way back).
export type TurnAnchor = 'bottom' | 'prompt'

/** Scope is in the key (desktop AGENTS.md): a reading preference is app-wide,
 *  not per connection or profile, so no suffix. */
export const TURN_ANCHOR_STORAGE_KEY = 'hermes.desktop.turnAnchor.v1'

export function normalizeTurnAnchor(value: null | string | undefined): TurnAnchor {
  return value === 'prompt' ? 'prompt' : 'bottom'
}

export const $turnAnchor = atom<TurnAnchor>(normalizeTurnAnchor(readKey(TURN_ANCHOR_STORAGE_KEY)))

export function setTurnAnchor(anchor: TurnAnchor): void {
  $turnAnchor.set(anchor)
  // The default stores NOTHING: a user who never touched the setting keeps
  // today's landing with no persisted state to migrate.
  writeKey(TURN_ANCHOR_STORAGE_KEY, anchor === 'bottom' ? null : anchor)
}

export interface TurnEndAnchorMetrics {
  anchor: TurnAnchor
  /** The viewport was still parked at the bottom when the turn ended. */
  atBottom: boolean
  maxScrollTop: number
  /** Viewport-relative top of the newest prompt's own (unpinned) position. */
  promptTop: null | number
  scrollTop: number
  viewportTop: number
}

/**
 * The scrollTop a just-finished turn settles on, or `null` to leave the
 * viewport alone. Pure, so the three guards that matter — the default anchor,
 * a reader who scrolled away mid-turn, and a transcript with no prompt to
 * anchor to — are provable without a DOM.
 */
export function turnEndScrollTop(metrics: TurnEndAnchorMetrics): null | number {
  if (metrics.anchor !== 'prompt' || !metrics.atBottom || metrics.promptTop === null) {
    return null
  }

  const target = metrics.scrollTop + (metrics.promptTop - metrics.viewportTop)

  return Math.min(Math.max(0, target), Math.max(0, metrics.maxScrollTop))
}

// Storage is scoped per profile with the same `.profile.<encoded>` suffix the
// app's other persisted session state uses (session.ts profileNavigationKey),
// so two profiles can never read or evict each other's reading positions.
const SCROLL_POS_KEY_BASE = 'hermes.desktop.threadScroll.v1'

export function threadScrollStorageKey(profile = $activeProfile.get()): string {
  return `${SCROLL_POS_KEY_BASE}.profile.${encodeURIComponent(normalizeProfileKey(profile))}${activeConnectionScopeSuffix()}`
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

  return (
    record.kind === 'offset' &&
    typeof record.fromBottom === 'number' &&
    Number.isFinite(record.fromBottom) &&
    record.fromBottom >= 0
  )
}

function loadPositions(storageKey: string): Record<string, ThreadScrollState> {
  const raw = readKey(storageKey)

  if (!raw) {
    return {}
  }

  try {
    const parsed = JSON.parse(raw) as unknown

    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      return {}
    }

    return Object.fromEntries(
      Object.entries(parsed as Record<string, unknown>).filter((entry): entry is [string, ThreadScrollState] =>
        isValidState(entry[1])
      )
    )
  } catch {
    return {}
  }
}

function persistPositions(storageKey: string, positions: Record<string, ThreadScrollState>) {
  const keys = Object.keys(positions)

  while (keys.length > THREAD_SCROLL_MEMORY_LIMIT) {
    delete positions[keys[0]!]
    keys.shift()
  }

  writeKey(storageKey, keys.length === 0 ? null : JSON.stringify(positions))
}

export function getThreadScrollPosition(
  sessionKey: string,
  storageKey = threadScrollStorageKey()
): ThreadScrollState | undefined {
  return loadPositions(storageKey)[sessionKey]
}

export function saveThreadScrollPosition(
  sessionKey: string,
  state: ThreadScrollState,
  storageKey = threadScrollStorageKey()
) {
  const positions = loadPositions(storageKey)

  // Delete then re-add to track recency (insertion order = LRU anchor).
  delete positions[sessionKey]
  positions[sessionKey] = state
  persistPositions(storageKey, positions)
}

export function clearThreadScrollPosition(sessionKey: string, storageKey = threadScrollStorageKey()) {
  const positions = loadPositions(storageKey)

  if (positions[sessionKey] === undefined) {
    return
  }

  delete positions[sessionKey]
  persistPositions(storageKey, positions)
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

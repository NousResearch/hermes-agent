import { atom } from 'nanostores'

import type { SessionInfo } from '@/types/hermes'

import { completeFlow } from './desktop-metrics'
import {
  $selectedStoredSessionId,
  $sessions,
  getSessionOwnerHint,
  sessionMatchesStoredId,
  sessionOwnerRouteFromRow
} from './session'

/** A switcher target is a ROW, not a bare id. Stored ids are only unique per
 *  profile (#92454), so the cross-profile `$sessions` list can hold two rows
 *  sharing one id; a surface that reduces a row to its id cannot then say
 *  WHICH chat it picked. Callers open the row through `openSessionFromRow`,
 *  which pins that row's own (connection, profile) as the resume owner, the
 *  same contract the Sessions sidebar row has. */
export type SwitcherTarget = SessionInfo

/** React key for a switcher row: (profile, id). A bare id collapses two twins
 *  into one key (#92454), so the HUD would reconcile one twin's row in the
 *  other's place. The two surfaces must key a row the same way they open it. */
export const switcherRowKey = (session: Pick<SessionInfo, 'id' | 'profile'>): string =>
  `${session.profile ?? ''}::${session.id}`

// Mac-style session switcher (^Tab). Quick tap jumps on keydown; the HUD opens
// only when Tab is held past REVEAL_MS or tapped again while Ctrl is down.

export const SWITCHER_REVEAL_MS = 220

export const $switcherOpen = atom(false)
export const $switcherSessions = atom<SessionInfo[]>([])
export const $switcherIndex = atom(0)

const wrap = (index: number, length: number): number => ((index % length) + length) % length

let pendingBrowse = false
let revealTimer: ReturnType<typeof setTimeout> | null = null
let tabHeld = false
let closedAt = 0

function clearRevealTimer(): void {
  if (revealTimer) {
    clearTimeout(revealTimer)
    revealTimer = null
  }
}

function revealOverlay(): void {
  pendingBrowse = false
  $switcherOpen.set(true)
}

function scheduleReveal(): void {
  clearRevealTimer()
  revealTimer = setTimeout(() => {
    revealTimer = null

    if (pendingBrowse && tabHeld) {
      revealOverlay()
    }
  }, SWITCHER_REVEAL_MS)
}

export function onSwitcherTabDown(): void {
  tabHeld = true
}

export function onSwitcherTabUp(): void {
  tabHeld = false

  if (!$switcherOpen.get()) {
    clearRevealTimer()
  }
}

// First Tab returns the row to jump to immediately; later Tabs move the
// highlight (Ctrl commits when the HUD is open).
export function openOrAdvanceSwitcher(direction: 1 | -1): null | SwitcherTarget {
  const sessions = $sessions.get()

  if (sessions.length < 2) {
    return null
  }

  if ($switcherOpen.get()) {
    const { length } = $switcherSessions.get()

    if (length) {
      $switcherIndex.set(wrap($switcherIndex.get() + direction, length))
    }

    return null
  }

  const selectedId = $selectedStoredSessionId.get()
  // Rows the selection could name, in list order. A stored id is not unique
  // across profiles (#92454), and it matches a row by lineage, not just the live
  // id — the same rule the sidebar resolves rows with (`sessionMatchesStoredId`).
  const candidates = selectedId === null ? [] : sessions.filter(session => sessionMatchesStoredId(session, selectedId))
  // The owner route recorded when a row was last opened (`requestSessionResume`
  // -> `setSessionOwnerHint`) names WHICH twin the user is on. It is absent when
  // this id was never opened here, or when both twins were — the app's selection
  // primitive is the id alone, so that case still falls back to the first match.
  const hint = selectedId === null ? undefined : getSessionOwnerHint(selectedId)
  const current = sessions.indexOf(
    (hint ? candidates.find(session => sessionOwnerRouteFromRow(session)?.profile === hint.profile) : undefined) ??
      candidates[0]
  )
  const start = current === -1 ? (direction === 1 ? -1 : 0) : current
  const nextIndex = wrap(start + direction, sessions.length)

  $switcherSessions.set(sessions)
  $switcherIndex.set(nextIndex)

  if (pendingBrowse) {
    clearRevealTimer()
    $switcherIndex.set(wrap($switcherIndex.get() + direction, sessions.length))
    revealOverlay()

    return null
  }

  pendingBrowse = true
  scheduleReveal()

  return sessions[nextIndex] ?? null
}

export const highlightedSession = (): null | SwitcherTarget => $switcherSessions.get()[$switcherIndex.get()] ?? null

export const slotSession = (slot: number): null | SwitcherTarget =>
  ($switcherOpen.get() || pendingBrowse ? $switcherSessions.get() : $sessions.get())[slot - 1] ?? null

export function closeSwitcher(): void {
  closedAt = Date.now()
  clearRevealTimer()
  pendingBrowse = false
  tabHeld = false
  $switcherOpen.set(false)
}

export function commitOnCtrlUp(): null | SwitcherTarget {
  clearRevealTimer()
  pendingBrowse = false

  if (!$switcherOpen.get()) {
    return null
  }

  const target = highlightedSession()

  if (target) {
    completeFlow('session_switcher')
  }

  closeSwitcher()

  return target
}

export const switcherJustClosed = (): boolean => Date.now() - closedAt < 400

export const switcherActive = (): boolean => $switcherOpen.get() || pendingBrowse

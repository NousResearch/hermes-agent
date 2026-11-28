import { atom } from 'nanostores'

import type { SessionInfo } from '@/types/hermes'

import { completeFlow } from './desktop-metrics'
import { $selectedStoredSessionId, $sessions } from './session'
import { $focusedStoredSessionId } from './session-focus'

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
// The physical key (`code`) of the keydown being dispatched, and of the one that
// stepped the switcher: only the step key's keyup ends the hold, so another key
// lifting mid-hold can't hide the HUD. A missing code on either side releases on
// any key, as before.
let keyDown = ''
let stepKey = ''

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

export function noteSwitcherKeyDown(code: string): void {
  keyDown = code
}

export function onSwitcherTabDown(): void {
  tabHeld = true
  stepKey = keyDown
}

export function onSwitcherTabUp(code = ''): void {
  if (code && stepKey && code !== stepKey) {
    return
  }

  tabHeld = false

  if (!$switcherOpen.get()) {
    clearRevealTimer()
  }
}

// First Tab returns a session id to jump to immediately; later Tabs move the
// highlight (Ctrl↑ commits when the HUD is open).
export function openOrAdvanceSwitcher(direction: 1 | -1): string | null {
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

  const current = sessions.findIndex(session => session.id === $selectedStoredSessionId.get())
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

  return sessions[nextIndex]?.id ?? null
}

export const highlightedSessionId = (): string | null => $switcherSessions.get()[$switcherIndex.get()]?.id ?? null

export const slotSessionId = (slot: number): string | null =>
  ($switcherOpen.get() || pendingBrowse ? $switcherSessions.get() : $sessions.get())[slot - 1]?.id ?? null

export function closeSwitcher(): void {
  closedAt = Date.now()
  clearRevealTimer()
  pendingBrowse = false
  tabHeld = false
  $switcherOpen.set(false)
}

export function commitOnCtrlUp(): string | null {
  clearRevealTimer()
  pendingBrowse = false

  if (!$switcherOpen.get()) {
    return null
  }

  const target = highlightedSessionId()

  if (target) {
    completeFlow('session_switcher')
  }

  closeSwitcher()

  return target
}

// "Go to last session": the chats this window showed, most recent first. The
// ^Tab switcher walks the sidebar's activity order, so when two chats are both
// busy it rarely lands on the one you just left; this order is yours alone.
// Pressing it again flips back, Alt-Tab style. Renderer-only presentation
// state: each window keeps its own trail and nothing persists.
const VISIT_TRAIL_LIMIT = 20
let visitTrail: string[] = []

export function noteSessionVisit(sessionId: null | string): void {
  if (sessionId && visitTrail[0] !== sessionId) {
    visitTrail = [sessionId, ...visitTrail.filter(id => id !== sessionId)].slice(0, VISIT_TRAIL_LIMIT)
  }
}

$focusedStoredSessionId.subscribe(noteSessionVisit)

// A chat that leaves the list (archived, deleted, another gateway) leaves the
// trail too. One the list never held (opened from search past its page) stays.
let listedIds = new Set<string>()

$sessions.subscribe(sessions => {
  const next = new Set(sessions.map(session => session.id))

  visitTrail = visitTrail.filter(id => next.has(id) || !listedIds.has(id))
  listedIds = next
})

/** The chat focused before the current one. From a blank draft this is the
 *  chat you just left. */
export const lastVisitedSessionId = (): string | null =>
  visitTrail.find(id => id !== $focusedStoredSessionId.get()) ?? null

export const switcherJustClosed = (): boolean => Date.now() - closedAt < 400

export const switcherActive = (): boolean => $switcherOpen.get() || pendingBrowse

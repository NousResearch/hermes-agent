import { atom } from 'nanostores'

import type { SessionInfo } from '@/types/hermes'

import { $activeGatewayProfile, normalizeProfileKey } from './profile'
import { $selectedStoredSessionId, $sessions } from './session'
import { makeSessionIdentity, type SessionIdentity } from './session-identity'

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

// First Tab returns a session id to jump to immediately; later Tabs move the
// highlight (Ctrl↑ commits when the HUD is open).
export function openOrAdvanceSwitcher(direction: 1 | -1): SessionIdentity | null {
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

  const activeProfile = normalizeProfileKey($activeGatewayProfile.get())

  const current = sessions.findIndex(
    session => session.id === $selectedStoredSessionId.get() && normalizeProfileKey(session.profile) === activeProfile
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

  const target = sessions[nextIndex]

  return target ? makeSessionIdentity(target.profile, target.id) : null
}

export const highlightedSession = (): SessionIdentity | null => {
  const target = $switcherSessions.get()[$switcherIndex.get()]

  return target ? makeSessionIdentity(target.profile, target.id) : null
}

export const slotSession = (slot: number): SessionIdentity | null => {
  const target = ($switcherOpen.get() || pendingBrowse ? $switcherSessions.get() : $sessions.get())[slot - 1]

  return target ? makeSessionIdentity(target.profile, target.id) : null
}

export function closeSwitcher(): void {
  closedAt = Date.now()
  clearRevealTimer()
  pendingBrowse = false
  tabHeld = false
  $switcherOpen.set(false)
}

export function commitOnCtrlUp(): SessionIdentity | null {
  clearRevealTimer()
  pendingBrowse = false

  if (!$switcherOpen.get()) {
    return null
  }

  const target = highlightedSession()
  closeSwitcher()

  return target
}

export const switcherJustClosed = (): boolean => Date.now() - closedAt < 400

export const switcherActive = (): boolean => $switcherOpen.get() || pendingBrowse

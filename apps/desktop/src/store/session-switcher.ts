import { atom } from 'nanostores'

import type { SessionInfo } from '@/types/hermes'

import { $activeGatewayProfile, normalizeProfileKey } from './profile'
import {
  $selectedStoredSessionId,
  $sessions,
  clearRequestedSessionResumeProfile,
  requestSessionResumeProfile
} from './session'

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

  const selectedProfile = normalizeProfileKey($activeGatewayProfile.get())

  const current = sessions.findIndex(
    session => session.id === $selectedStoredSessionId.get() && normalizeProfileKey(session.profile) === selectedProfile
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

  if (target) {
    requestSessionResumeProfile(target.id, target.profile)
  }

  return target?.id ?? null
}

export const highlightedSessionId = (): string | null => $switcherSessions.get()[$switcherIndex.get()]?.id ?? null

export const slotSessionId = (slot: number): string | null =>
  requestedSessionId(($switcherOpen.get() || pendingBrowse ? $switcherSessions.get() : $sessions.get())[slot - 1])

function requestedSessionId(session: SessionInfo | undefined): string | null {
  if (!session) {
    return null
  }

  requestSessionResumeProfile(session.id, session.profile)

  return session.id
}

export function closeSwitcher(): void {
  closedAt = Date.now()
  clearRevealTimer()
  pendingBrowse = false
  tabHeld = false
  $switcherOpen.set(false)
  clearRequestedSessionResumeProfile()
}

export function commitOnCtrlUp(): string | null {
  clearRevealTimer()
  pendingBrowse = false

  if (!$switcherOpen.get()) {
    return null
  }

  const target = $switcherSessions.get()[$switcherIndex.get()]
  closeSwitcher()

  return requestedSessionId(target)
}

export const switcherJustClosed = (): boolean => Date.now() - closedAt < 400

export const switcherActive = (): boolean => $switcherOpen.get() || pendingBrowse

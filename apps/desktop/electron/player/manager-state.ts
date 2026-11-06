import type { BrowserWindow, Rectangle, WebContentsView } from 'electron'

import type { PlayerAgentScope } from '../../src/lib/player-context-contract'
import type { PlayerSnapshot, PlayerSource } from '../../src/lib/player-contract'

import type { GuestBoundsTracking } from './float-guest-bounds'
import type { FixtureOrigin } from './policy'

export type { FixtureOrigin } from './policy'

export type Slot = {
  owner: BrowserWindow | null
  ownerId: number
  guest: WebContentsView
  source: PlayerSource
  presentation: 'docked' | 'floating'
  floatWindow: BrowserWindow | null
  floatBoundsTracking: GuestBoundsTracking | null
  floatBounds: Rectangle | null
  fullscreen: boolean
  fullscreenOrigin: 'docked' | 'floating' | null
  pinned: boolean
  pinStatus: PlayerSnapshot['pinStatus']
  pinMode: PlayerSnapshot['pinMode']
  visible: boolean
  hasBounds: boolean
  bounds: { x: number; y: number; width: number; height: number }
  boundsRevision: number
  status: PlayerSnapshot['status']
  currentTime: number | null
  duration: number | null
  paused: boolean | null
  error: string | null
  runtime: PlayerSnapshot['runtime']
  mediaPoll: ReturnType<typeof setInterval> | null
  mediaPollInFlight: boolean
  navigationGeneration: number
  muted: boolean | null
  authWindows: Set<BrowserWindow>
  closeInProgress: boolean
}

export type AwarenessLink = {
  ownerId: number
  linkId: string
  scope: PlayerAgentScope
}

export function safePageText(value: unknown, maxLength: number): { text: string; trust: 'untrusted-page' } | null {
  if (typeof value !== 'string') {
    return null
  }

  const text = [...value]
    .map(character => {
      const code = character.charCodeAt(0)

      return code <= 0x1f || code === 0x7f ? ' ' : character
    })
    .join('')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, maxLength)

  return text ? { text, trust: 'untrusted-page' } : null
}

export function validSeconds(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null
}

export function fixtureOrigin(fixtureUrl?: string): FixtureOrigin | undefined {
  if (!fixtureUrl) {
    return undefined
  }

  try {
    const url = new URL(fixtureUrl)

    if (url.protocol !== 'http:' || url.hostname !== '127.0.0.1' || !url.port) {
      return undefined
    }

    return { hostname: '127.0.0.1', port: url.port }
  } catch {
    return undefined
  }
}

export function initialSnapshot(): PlayerSnapshot {
  return {
    status: 'closed',
    source: null,
    presentation: 'docked',
    pinned: false,
    pinStatus: 'requested',
    pinMode: 'topmost',
    fullscreen: false,
    currentTime: null,
    duration: null,
    paused: null,
    muted: null,
    canGoBack: false,
    canGoForward: false,
    awarenessLinked: false,
    safeOrigin: null,
    guestId: null,
    runtime: { state: 'starting' },
    error: null
  }
}

export function redactError(error: unknown): string {
  const name = error instanceof Error ? error.name : 'Error'

  return /^[A-Za-z][A-Za-z0-9]{0,39}$/.test(name) ? name : 'Error'
}

export function isDocked(slot: Slot): boolean {
  return slot.presentation === 'docked'
}

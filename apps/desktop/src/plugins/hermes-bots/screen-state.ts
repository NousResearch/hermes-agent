/**
 * Per-bot Bot Screen cache: backend truth (`display.status`) plus the last
 * `display.lease` event, keyed by the bot's roster identity. Renderer-owned
 * cache of backend state — never the authority.
 */

import { atom } from 'nanostores'

import { botSelectionKey } from './data'
import type { DisplayLease, DisplayStatus, ScreenViewer } from './screen-connection'
import type { RosterRow } from './types'

export interface BotScreenState {
  status: DisplayStatus | null
  lease: DisplayLease | null
  /** This window's server-minted identity for the bot's current attach; null until the pane observes. */
  viewer: ScreenViewer | null
  /** The bot's Hermes has no `display.*` methods (older backend): nothing to check, ever. */
  unavailable?: boolean
}

export const $screenState = atom<Record<string, BotScreenState>>({})

export function screenStateFor(all: Record<string, BotScreenState>, bot: RosterRow): BotScreenState | null {
  return all[botSelectionKey(bot)] ?? null
}

/** A lease whose epoch is below the one we hold is a slower response about the
 *  past (a `display.status` reply overtaken by a `display.lease` event). Payloads
 *  without an epoch — older backends — are always applied. */
function isOlderLease(prev: DisplayLease | null | undefined, next: DisplayLease): boolean {
  return typeof next.epoch === 'number' && typeof prev?.epoch === 'number' && next.epoch < prev.epoch
}

export function setScreenStatus(bot: RosterRow, status: DisplayStatus): void {
  const key = botSelectionKey(bot)
  const current = $screenState.get()
  const prev = current[key]
  const lease = status.lease && !isOlderLease(prev?.lease, status.lease) ? status.lease : (prev?.lease ?? null)
  $screenState.set({ ...current, [key]: { status, lease, viewer: prev?.viewer ?? null } })
}

/** `display.status` answered method-not-found: remember it so no surface keeps "checking". */
export function setScreenUnavailable(bot: RosterRow): void {
  const key = botSelectionKey(bot)
  const current = $screenState.get()
  const prev = current[key]

  if (prev?.unavailable) {
    return
  }

  $screenState.set({ ...current, [key]: { status: null, lease: null, viewer: null, unavailable: true } })
}

export function setScreenLease(bot: RosterRow, lease: DisplayLease): void {
  const key = botSelectionKey(bot)
  const current = $screenState.get()
  const prev = current[key]

  if (isOlderLease(prev?.lease, lease)) {
    return
  }

  if (
    prev?.lease &&
    prev.lease.holder === lease.holder &&
    prev.lease.viewer_id === lease.viewer_id &&
    prev.lease.viewer_hash === lease.viewer_hash &&
    prev.lease.pending_handoff === lease.pending_handoff
  ) {
    return
  }

  $screenState.set({ ...current, [key]: { status: prev?.status ?? null, lease, viewer: prev?.viewer ?? null } })
}

/** Record the identity `display.observe` minted for this window's attach to `bot`. */
export function setScreenViewer(bot: RosterRow, viewer: ScreenViewer | null): void {
  const key = botSelectionKey(bot)
  const current = $screenState.get()
  const prev = current[key]

  if ((prev?.viewer ?? null) === viewer) {
    return
  }

  $screenState.set({ ...current, [key]: { status: prev?.status ?? null, lease: prev?.lease ?? null, viewer } })
}

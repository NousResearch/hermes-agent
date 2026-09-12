/**
 * Per-bot Bot Screen cache: backend truth (`display.status`) plus the last
 * `display.lease` event, keyed by the bot's roster identity. Renderer-owned
 * cache of backend state — never the authority.
 */

import { atom } from 'nanostores'

import { botSelectionKey } from './data'
import type { DisplayLease, DisplayStatus } from './screen-connection'
import type { RosterRow } from './types'

export interface BotScreenState {
  status: DisplayStatus | null
  lease: DisplayLease | null
}

export const $screenState = atom<Record<string, BotScreenState>>({})

export function screenStateFor(all: Record<string, BotScreenState>, bot: RosterRow): BotScreenState | null {
  return all[botSelectionKey(bot)] ?? null
}

export function setScreenStatus(bot: RosterRow, status: DisplayStatus): void {
  const key = botSelectionKey(bot)
  const current = $screenState.get()
  $screenState.set({ ...current, [key]: { status, lease: status.lease ?? current[key]?.lease ?? null } })
}

export function setScreenLease(bot: RosterRow, lease: DisplayLease): void {
  const key = botSelectionKey(bot)
  const current = $screenState.get()
  const prev = current[key]

  if (prev?.lease && prev.lease.holder === lease.holder && prev.lease.viewer_id === lease.viewer_id && prev.lease.pending_handoff === lease.pending_handoff) {
    return
  }

  $screenState.set({ ...current, [key]: { status: prev?.status ?? null, lease } })
}

import { atom } from 'nanostores'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $activeSessionId, $selectedStoredSessionId } from './session'
import { runtimeIdForStoredSession } from './session-runtime-binding'
import { $sessionStates, $sessionTiles } from './session-states'

vi.mock('./session', () => ({
  $activeSessionId: atom<null | string>(null),
  $selectedStoredSessionId: atom<null | string>(null)
}))
vi.mock('./session-states', () => ({
  $sessionStates: atom<Record<string, { storedSessionId: string }>>({}),
  $sessionTiles: atom<Array<{ storedSessionId: string; runtimeId: string }>>([])
}))

beforeEach(() => {
  $activeSessionId.set('foreground-runtime')
  $selectedStoredSessionId.set('foreground-row')
  $sessionStates.set({})
  $sessionTiles.set([])
})

describe('captured conversation runtime binding', () => {
  it('resolves the selected row to its active runtime', () => {
    expect(runtimeIdForStoredSession('foreground-row')).toBe('foreground-runtime')
  })

  it('resolves a background tile without borrowing the active runtime', () => {
    $sessionTiles.set([{ storedSessionId: 'background-row', runtimeId: 'background-runtime' }])
    expect(runtimeIdForStoredSession('background-row')).toBe('background-runtime')
  })

  it('resolves a mirrored conversation even when no tile is mounted', () => {
    $sessionStates.set({ 'background-runtime': { storedSessionId: 'background-row' } } as unknown as Parameters<
      typeof $sessionStates.set
    >[0])
    expect(runtimeIdForStoredSession('background-row')).toBe('background-runtime')
  })

  it('returns no runtime for an unrelated stored row', () => {
    expect(runtimeIdForStoredSession('not-open')).toBeNull()
  })

  it('still resolves a tile when the selected runtime is not hydrated yet', () => {
    $activeSessionId.set(null)
    $sessionTiles.set([{ storedSessionId: 'foreground-row', runtimeId: 'tile-runtime' }])
    expect(runtimeIdForStoredSession('foreground-row')).toBe('tile-runtime')
  })
})

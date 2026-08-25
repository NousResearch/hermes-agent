import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  $selectedStoredSessionId,
  _resetSessionOwnerHintsForTests,
  setActiveSessionId,
  setSessionOwnerHint
} from '@/store/session'

import {
  $sessionTiles,
  _resetSessionOwnerHoldsForTests,
  foregroundSessionScopes,
  holdSessionOwnerUntilForeground,
  recordSessionEventScope,
  releaseSessionOwnerHold
} from './session-states'

// A routed session.create returns a stored id on the owner's socket, but the
// surface that will pin that socket (the selected primary thread, or a tile)
// is published later and asynchronously. The hold names the owner in
// foregroundSessionScopes — the gateway keep-set — from the moment the create
// returns until the foreground publication takes over, the caller releases
// it, or a bounded TTL expires. Nothing latches.

afterEach(() => {
  $sessionTiles.set([])
  setActiveSessionId(null)
  $selectedStoredSessionId.set(null)
  _resetSessionOwnerHoldsForTests()
  _resetSessionOwnerHintsForTests({ storage: true })
  vi.useRealTimers()
})

describe('foregroundSessionScopes: owner hold across the create → foreground gap', () => {
  const omar = { connectionId: 'local', mode: 'local' as const, profile: 'omar' }

  it('names the owner from the moment a routed create returns, before anything is selected or tiled', () => {
    holdSessionOwnerUntilForeground('stored-fresh', omar)

    expect(foregroundSessionScopes()).toEqual(new Set(['conn:local::omar']))
  })

  it('retires once the foreground publication covers it (selected primary thread / mounted tile)', () => {
    holdSessionOwnerUntilForeground('stored-fresh', omar)
    setSessionOwnerHint('stored-fresh', omar)

    // Selected, but the runtime's event scope is not known yet: the hold is
    // still the only thing naming the owner socket, so it stays.
    $selectedStoredSessionId.set('stored-fresh')
    expect(foregroundSessionScopes()).toEqual(new Set(['conn:local::omar']))

    // The first event from the owner socket records the runtime's scope; the
    // selected-thread rung now covers it and the hold retires for good.
    recordSessionEventScope({ connectionId: 'local', profile: 'omar', session_id: 'rt-fresh' })
    setActiveSessionId('rt-fresh')
    expect(foregroundSessionScopes()).toEqual(new Set(['conn:local::omar']))
    setActiveSessionId(null)
    $selectedStoredSessionId.set(null)
    _resetSessionOwnerHintsForTests()
    expect(foregroundSessionScopes()).toEqual(new Set())

    holdSessionOwnerUntilForeground('stored-tile', { connectionId: 'homelab', profile: 'bot' })
    $sessionTiles.set([{ ownerRoute: { connectionId: 'homelab', profile: 'bot' }, storedSessionId: 'stored-tile' }])
    // Covered by the tile's own route rung now; the hold retired.
    expect(foregroundSessionScopes()).toEqual(new Set(['conn:homelab::bot']))
    $sessionTiles.set([])
    expect(foregroundSessionScopes()).toEqual(new Set())
  })

  it('is released explicitly by the caller (failed create / drift close) and expires on its own', () => {
    vi.useFakeTimers()

    const release = holdSessionOwnerUntilForeground('stored-a', omar)
    holdSessionOwnerUntilForeground('stored-b', { connectionId: 'homelab', profile: 'worker' })

    release()
    expect(foregroundSessionScopes()).toEqual(new Set(['conn:homelab::worker']))

    releaseSessionOwnerHold('stored-b')
    expect(foregroundSessionScopes()).toEqual(new Set())

    holdSessionOwnerUntilForeground('stored-c', omar)
    vi.advanceTimersByTime(60_000 + 1)
    expect(foregroundSessionScopes()).toEqual(new Set())
  })

  it('ignores blank ids, null owners and profile-only owners map to the legacy pool key', () => {
    holdSessionOwnerUntilForeground('  ', omar)
    holdSessionOwnerUntilForeground('stored-null', null)
    holdSessionOwnerUntilForeground('stored-legacy', 'research')

    expect(foregroundSessionScopes()).toEqual(new Set(['research']))
  })
})

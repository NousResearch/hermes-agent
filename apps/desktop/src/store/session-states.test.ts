import { afterEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import type * as ProfileStore from '@/store/profile'
import { $activeGatewayProfile } from '@/store/profile'
import { $activeSessionId, $selectedStoredSessionId } from '@/store/session'

const ensureGatewayProfile = vi.hoisted(() => vi.fn())

vi.mock('@/store/profile', async importOriginal => {
  const actual = await importOriginal<typeof ProfileStore>()

  ensureGatewayProfile.mockImplementation(async (profile: string) => {
    actual.$activeGatewayProfile.set(profile)
  })

  return { ...actual, ensureGatewayProfile }
})

import { $sessionStates, $sessionTiles, discardSessionTile, focusOpenSession, openSessionTile } from './session-states'

afterEach(() => {
  ensureGatewayProfile.mockClear()
  discardSessionTile('shared-id', 'default')
  discardSessionTile('shared-id', 'work')
  $activeGatewayProfile.set('default')
  $activeSessionId.set(null)
  $selectedStoredSessionId.set(null)
  $sessionStates.set({})
  $sessionTiles.set([])
})

describe('profile-aware session tiles', () => {
  it('switches to the owner profile before opening a colliding stored id', async () => {
    $activeGatewayProfile.set('default')
    $activeSessionId.set('runtime-default')
    $selectedStoredSessionId.set('shared-id')
    $sessionStates.set({
      'runtime-default': createClientSessionState('shared-id', [], 'default')
    })

    await openSessionTile('shared-id', 'center', undefined, undefined, 'work')

    expect(ensureGatewayProfile).toHaveBeenCalledWith('work')
    expect($activeGatewayProfile.get()).toBe('work')
    expect($sessionTiles.get()).toEqual([expect.objectContaining({ storedSessionId: 'shared-id' })])
  })

  it('discards only the target profile tile when stored ids collide', async () => {
    $activeGatewayProfile.set('default')
    await openSessionTile('shared-id', 'right', undefined, undefined, 'default')
    await openSessionTile('shared-id', 'right', undefined, undefined, 'work')

    $activeGatewayProfile.set('default')
    expect($sessionTiles.get()).toEqual([expect.objectContaining({ storedSessionId: 'shared-id' })])

    discardSessionTile('shared-id', 'work')

    expect($sessionTiles.get()).toEqual([expect.objectContaining({ storedSessionId: 'shared-id' })])

    $activeGatewayProfile.set('work')
    expect($sessionTiles.get()).toEqual([])
  })

  it('does not focus a colliding id from another profile', () => {
    $activeGatewayProfile.set('default')
    $selectedStoredSessionId.set('shared-id')

    expect(focusOpenSession('shared-id', 'work')).toBe(false)
  })
})

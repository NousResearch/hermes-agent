import { act, cleanup, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { clearSessionDraft, stashSessionDraft, takeSessionDraft } from '@/store/composer'
import {
  $sessionResumeRequest,
  _resetSessionOwnerHintsForTests,
  setSessionOwnerHint
} from '@/store/session'
import {
  $sessionTiles,
  sessionTileDelegate,
  setSessionTileDelegate
} from '@/store/session-states'
import type { RpcEvent } from '@/types/hermes'

import { renderMessageStream } from './test-harness'

const replayGap = (
  sessionId: string,
  owner: { connectionId: string; profile: string }
): RpcEvent => ({
  ...owner,
  payload: { latest_seq: 41, replay_epoch: 'epoch-next' },
  session_id: sessionId,
  type: 'session.replay_gap'
}) as RpcEvent

const inertDelegate = () => ({
  archiveSession: vi.fn(async () => undefined),
  branchSession: vi.fn(async () => undefined),
  deleteSession: vi.fn(async () => undefined),
  executeSlash: vi.fn(async () => undefined),
  interruptSession: vi.fn(async () => undefined),
  resumeTile: vi.fn(async () => 'runtime-unused'),
  submitToSession: vi.fn(async () => undefined),
  updateSession: vi.fn(state => state)
})

describe('session.replay_gap recovery', () => {
  beforeEach(() => {
    $sessionResumeRequest.set(null)
    $sessionTiles.set([])
    _resetSessionOwnerHintsForTests()
    setSessionTileDelegate(inertDelegate() as never)
  })

  afterEach(() => {
    cleanup()
    clearSessionDraft('stored-active')
    clearSessionDraft('stored-tile')
    $sessionResumeRequest.set(null)
    $sessionTiles.set([])
    _resetSessionOwnerHintsForTests()
  })

  it('requests an authoritative resume for the affected active route without losing owner scope or its draft', () => {
    const ownerRoute = {
      connectionId: 'remote-a',
      mode: 'remote' as const,
      profile: 'desktop-profile',
      targetProfile: 'backend-profile'
    }

    const state = createClientSessionState('stored-active')

    setSessionOwnerHint('stored-active', ownerRoute)
    stashSessionDraft('stored-active', 'keep typing', [])

    const stream = renderMessageStream('runtime-active', {
      states: new Map([['runtime-active', state]])
    })

    act(() => stream.handleEvent(replayGap('runtime-active', ownerRoute)))

    expect($sessionResumeRequest.get()).toMatchObject({
      authoritativeSnapshot: true,
      ownerRoute,
      sessionId: 'stored-active'
    })
    expect(takeSessionDraft('stored-active').text).toBe('keep typing')
  })

  it('re-resumes the affected mounted tile for a gateway snapshot while retaining its owner and draft', async () => {
    const ownerRoute = {
      connectionId: 'remote-b',
      mode: 'remote' as const,
      profile: 'tile-profile',
      targetProfile: 'backend-tile-profile'
    }

    const resumeTile = vi.fn(async () => 'runtime-tile')
    const delegate = { ...inertDelegate(), resumeTile }
    const state = createClientSessionState('stored-tile')

    state.messages = [{
      id: 'pending-user',
      parts: [{ type: 'text', text: 'local pending prompt' }],
      pending: true,
      role: 'user'
    }]
    $sessionTiles.set([{ ownerRoute, runtimeId: 'runtime-tile', storedSessionId: 'stored-tile' }] as never)
    setSessionTileDelegate(delegate as never)
    stashSessionDraft('stored-tile', 'unsent tile draft', [])

    const stream = renderMessageStream('runtime-other', {
      states: new Map([['runtime-tile', state]])
    })

    act(() => stream.handleEvent(replayGap('runtime-tile', ownerRoute)))

    await waitFor(() =>
      expect(resumeTile).toHaveBeenCalledWith('stored-tile', { authoritativeSnapshot: true })
    )
    expect($sessionTiles.get()[0]?.ownerRoute).toEqual(ownerRoute)
    expect(takeSessionDraft('stored-tile').text).toBe('unsent tile draft')
    expect(sessionTileDelegate()).toBe(delegate)
  })

  it('ignores a colliding runtime id delivered by a different exact owner', () => {
    const ownerRoute = { connectionId: 'remote-a', profile: 'default' }
    const state = createClientSessionState('stored-active')
    const resumeTile = vi.fn(async () => 'runtime-collision')

    setSessionOwnerHint('stored-active', ownerRoute)
    $sessionTiles.set([
      { ownerRoute, runtimeId: 'runtime-collision', storedSessionId: 'stored-tile' }
    ] as never)
    setSessionTileDelegate({ ...inertDelegate(), resumeTile } as never)

    const stream = renderMessageStream('runtime-collision', {
      states: new Map([['runtime-collision', state]])
    })

    act(() =>
      stream.handleEvent(replayGap('runtime-collision', { connectionId: 'remote-b', profile: 'default' }))
    )

    expect($sessionResumeRequest.get()).toBeNull()
    expect(resumeTile).not.toHaveBeenCalled()
  })
})

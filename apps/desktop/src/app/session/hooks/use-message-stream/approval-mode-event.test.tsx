import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $approvalModes, approvalModeForProfile } from '@/store/approval-mode'
import { $petUnread } from '@/store/pet'
import { $petLiveSessions, resetPetLiveSessions } from '@/store/pet-live-session'
import { $activeGatewayProfile } from '@/store/profile'
import { $currentModel, setCurrentModel } from '@/store/session'
import { getProfileSessionValue, setProfileSessionValue } from '@/store/session-identity'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

const ACTIVE_SID = 'session-active'
let handleEvent: ((event: RpcEvent) => void) | null = null
let cachedStates: Map<string, ClientSessionState> | null = null

function Harness() {
  const activeSessionIdRef = useRef<string | null>(ACTIVE_SID)
  const sessionStateByRuntimeIdRef = useRef(new Map<string, ClientSessionState>())
  const queryClientRef = useRef(new QueryClient())

  const stream = useMessageStream({
    activeSessionIdRef,
    hydrateFromStoredSession: vi.fn(async () => undefined),
    queryClient: queryClientRef.current,
    refreshHermesConfig: vi.fn(async () => undefined),
    refreshSessions: vi.fn(async () => undefined),
    sessionStateByRuntimeIdRef,
    updateSessionState: (sessionId, updater, storedSessionId, profile = 'default') => {
      const current =
        getProfileSessionValue(sessionStateByRuntimeIdRef.current, profile, sessionId) ??
        createClientSessionState(storedSessionId ?? null, [], profile)

      const next = updater(current)
      setProfileSessionValue(sessionStateByRuntimeIdRef.current, profile, sessionId, next)

      return next
    }
  })

  cachedStates = sessionStateByRuntimeIdRef.current

  useEffect(() => {
    handleEvent = stream.handleGatewayEvent
  }, [stream.handleGatewayEvent])

  return null
}

async function mountStream() {
  render(<Harness />)
  await waitFor(() => expect(handleEvent).not.toBeNull())
}

describe('live session.info approval mode reconciliation', () => {
  beforeEach(() => {
    handleEvent = null
    cachedStates = null
    $approvalModes.set({})
    $activeGatewayProfile.set('work')
    setCurrentModel('foreground-model')
    resetPetLiveSessions()
    $petUnread.set(false)
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    resetPetLiveSessions()
    $petUnread.set(false)
  })

  it('reconciles an active-session event under its source gateway profile', async () => {
    await mountStream()

    act(() =>
      handleEvent!({
        payload: { approval_mode: 'off' },
        profile: 'work',
        session_id: ACTIVE_SID,
        type: 'session.info'
      })
    )

    expect(approvalModeForProfile('work')).toBe('off')
    expect(approvalModeForProfile('default')).toBe('smart')
  })

  it('ignores stale session.info from a non-active session on the active gateway', async () => {
    await mountStream()

    act(() =>
      handleEvent!({
        payload: { approval_mode: 'off' },
        profile: 'work',
        session_id: 'session-stale',
        type: 'session.info'
      })
    )

    expect(approvalModeForProfile('work')).toBe('smart')
  })

  it('does not cache an event under a different active profile when its source profile is absent', async () => {
    await mountStream()
    $activeGatewayProfile.set('personal')

    act(() =>
      handleEvent!({ payload: { approval_mode: 'off' }, session_id: ACTIVE_SID, type: 'session.info' })
    )

    expect(approvalModeForProfile('personal')).toBe('smart')
    expect(approvalModeForProfile('work')).toBe('smart')
  })

  it('does not classify the same runtime id from another profile as active', async () => {
    await mountStream()
    $activeGatewayProfile.set('default')

    act(() =>
      handleEvent!({
        payload: { model: 'wrong-profile-model' },
        profile: 'work',
        session_id: ACTIVE_SID,
        type: 'session.info'
      })
    )

    expect($currentModel.get()).toBe('foreground-model')
  })

  it('routes the same runtime id in two profiles to separate cache entries', async () => {
    await mountStream()

    act(() => {
      handleEvent!({
        payload: { cwd: '/default' },
        profile: 'default',
        session_id: ACTIVE_SID,
        type: 'session.info'
      })
      handleEvent!({
        payload: { cwd: '/work' },
        profile: 'work',
        session_id: ACTIVE_SID,
        type: 'session.info'
      })
    })

    expect(cachedStates?.size).toBe(2)
    expect(getProfileSessionValue(cachedStates!, 'default', ACTIVE_SID)?.cwd).toBe('/default')
    expect(getProfileSessionValue(cachedStates!, 'work', ACTIVE_SID)?.cwd).toBe('/work')
  })

  it('completes one same-runtime profile without mutating the other', async () => {
    await mountStream()

    act(() => {
      handleEvent!({ profile: 'default', session_id: ACTIVE_SID, type: 'message.start' })
      handleEvent!({ profile: 'work', session_id: ACTIVE_SID, type: 'message.start' })
      handleEvent!({ payload: { text: 'done' }, profile: 'work', session_id: ACTIVE_SID, type: 'message.complete' })
    })

    expect(getProfileSessionValue(cachedStates!, 'default', ACTIVE_SID)?.busy).toBe(true)
    expect(getProfileSessionValue(cachedStates!, 'work', ACTIVE_SID)?.busy).toBe(false)
  })

  it('projects only bounded direct activity metadata and clears matching tool activity', async () => {
    await mountStream()

    act(() => {
      handleEvent!({
        payload: { args: { token: 'secret' }, name: `  ${'x'.repeat(500)}  `, output: 'private output' },
        profile: 'work',
        session_id: ACTIVE_SID,
        type: 'tool.start'
      })
    })

    expect($petLiveSessions.get()).toEqual([
      expect.objectContaining({ activityKind: 'tool', activityName: 'x'.repeat(120), profile: 'work' })
    ])
    expect(JSON.stringify($petLiveSessions.get())).not.toContain('secret')
    expect(JSON.stringify($petLiveSessions.get())).not.toContain('private output')

    act(() => {
      handleEvent!({
        payload: { name: `  ${'x'.repeat(500)}  ` },
        profile: 'work',
        session_id: ACTIVE_SID,
        type: 'tool.complete'
      })
    })

    expect($petLiveSessions.get()[0]).toEqual(expect.objectContaining({ activityKind: null, activityName: null }))
  })

  it('marks background completion and errors as exact outcomes and global unread without profile contamination', async () => {
    vi.spyOn(document, 'hasFocus').mockReturnValue(false)
    await mountStream()

    act(() => {
      handleEvent!({ payload: { text: 'private reasoning' }, profile: 'default', session_id: ACTIVE_SID, type: 'reasoning.delta' })
      handleEvent!({ payload: { text: 'done' }, profile: 'default', session_id: ACTIVE_SID, type: 'message.complete' })
      handleEvent!({ payload: { message: 'internal failure detail' }, profile: 'work', session_id: ACTIVE_SID, type: 'error' })
    })

    expect($petLiveSessions.get()).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ profile: 'default', runtimeSessionId: ACTIVE_SID, outcome: 'done' }),
        expect.objectContaining({ profile: 'work', runtimeSessionId: ACTIVE_SID, outcome: 'failed' })
      ])
    )
    expect(JSON.stringify($petLiveSessions.get())).not.toContain('private reasoning')
    expect(JSON.stringify($petLiveSessions.get())).not.toContain('internal failure detail')
    expect($petUnread.get()).toBe(true)
  })
})

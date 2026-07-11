import { act, cleanup, render, waitFor } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'
import { $activeGatewayProfile, $freshSessionRequest, $profileSwitchTarget } from '@/store/profile'
import { setSessions } from '@/store/session'

import { useProfileSessionRestore } from './use-profile-session-restore'

function storedSession(overrides: Partial<SessionInfo> = {}): SessionInfo {
  return {
    ended_at: null,
    id: 'stored-1',
    input_tokens: 0,
    is_active: false,
    last_active: 1,
    message_count: 1,
    model: null,
    output_tokens: 0,
    preview: null,
    profile: 'default',
    source: 'desktop',
    started_at: 1,
    title: 'stored',
    tool_call_count: 0,
    ...overrides
  }
}

function Harness({
  resumeSession,
  selectedStoredSessionId,
  selectedStoredSessionIdRef,
  startFreshSessionDraft
}: {
  resumeSession: (id: string, replaceRoute?: boolean) => Promise<void>
  selectedStoredSessionId: string | null
  selectedStoredSessionIdRef: MutableRefObject<string | null>
  startFreshSessionDraft: () => void
}) {
  useProfileSessionRestore({
    resumeSession,
    selectedStoredSessionId,
    selectedStoredSessionIdRef,
    startFreshSessionDraft
  })

  return null
}

function requestFreshForProfile(profile: string): void {
  act(() => {
    $profileSwitchTarget.set(profile)
    $freshSessionRequest.set($freshSessionRequest.get() + 1)
  })
}

describe('useProfileSessionRestore', () => {
  afterEach(() => {
    cleanup()
    $activeGatewayProfile.set('default')
    $freshSessionRequest.set(0)
    $profileSwitchTarget.set(null)
    setSessions([])
    vi.restoreAllMocks()
  })

  it('restores the last opened session for a profile switch', async () => {
    const resumeSession = vi.fn(async () => undefined)
    const startFreshSessionDraft = vi.fn()
    const selectedStoredSessionIdRef: MutableRefObject<string | null> = { current: 'session-a' }

    setSessions([
      storedSession({ id: 'session-a', profile: 'default', title: 'Default chat' }),
      storedSession({ id: 'session-b', profile: 'coder', title: 'Coder chat' })
    ])

    const view = render(
      <Harness
        resumeSession={resumeSession}
        selectedStoredSessionId="session-a"
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
        startFreshSessionDraft={startFreshSessionDraft}
      />
    )

    act(() => {
      $activeGatewayProfile.set('coder')
      selectedStoredSessionIdRef.current = 'session-b'
    })
    view.rerender(
      <Harness
        resumeSession={resumeSession}
        selectedStoredSessionId="session-b"
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
        startFreshSessionDraft={startFreshSessionDraft}
      />
    )

    act(() => {
      $activeGatewayProfile.set('default')
      selectedStoredSessionIdRef.current = 'session-a'
    })
    view.rerender(
      <Harness
        resumeSession={resumeSession}
        selectedStoredSessionId="session-a"
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
        startFreshSessionDraft={startFreshSessionDraft}
      />
    )

    requestFreshForProfile('coder')

    await waitFor(() => expect(resumeSession).toHaveBeenCalledWith('session-b', true))
    expect(startFreshSessionDraft).not.toHaveBeenCalled()
    expect($profileSwitchTarget.get()).toBeNull()
  })

  it('starts a fresh draft when the target profile has no remembered session', async () => {
    const resumeSession = vi.fn(async () => undefined)
    const startFreshSessionDraft = vi.fn()
    const selectedStoredSessionIdRef: MutableRefObject<string | null> = { current: 'session-a' }

    setSessions([storedSession({ id: 'session-a', profile: 'default' })])

    render(
      <Harness
        resumeSession={resumeSession}
        selectedStoredSessionId="session-a"
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
        startFreshSessionDraft={startFreshSessionDraft}
      />
    )

    requestFreshForProfile('coder')

    await waitFor(() => expect(startFreshSessionDraft).toHaveBeenCalledTimes(1))
    expect(resumeSession).not.toHaveBeenCalled()
    expect($profileSwitchTarget.get()).toBeNull()
  })
})

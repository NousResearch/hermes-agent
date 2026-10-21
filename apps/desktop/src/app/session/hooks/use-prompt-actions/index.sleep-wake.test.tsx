import { JsonRpcGatewayError } from '@hermes/shared'
import { cleanup, render, waitFor } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getLatestSessionMessages, getSession } from '@/hermes'
import { textPart } from '@/lib/chat-messages'
import { $busy, $messages, $turnStartedAt, setMessages, setSessions } from '@/store/session'

import { actRender, Harness, type HarnessHandle, RUNTIME_SESSION_ID, sessionInfo } from './index-test-utils'
import { clearSingleFlightSessionResumeState } from './single-flight-resume'

// Suites in this file reuse the same stored-id constants. The module-level
// single-flight resume map (and drift-recovery cache) would otherwise leak a
// never-settling in-flight promise from one test into the next.
beforeEach(() => {
  clearSingleFlightSessionResumeState()
  window.localStorage.removeItem('hermes.desktop.preparedSubmissions.v1')
  // Queue mutations build on the persisted map, not the atom — a queue an
  // earlier test left in storage would otherwise sit ahead of this test's send.
  window.localStorage.removeItem('hermes.desktop.composerQueue.v1')
  vi.mocked(getLatestSessionMessages).mockReset()
  vi.mocked(getLatestSessionMessages).mockImplementation(async () => ({ messages: [], session_id: 'session' }))
})

vi.mock('@/hermes', () => ({
  getLatestSessionMessages: vi.fn(async () => ({ messages: [], session_id: 'session' })),
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  getSession: vi.fn(),
  PROMPT_SUBMIT_REQUEST_TIMEOUT_MS: 1_800_000,
  setApiRequestProfile: vi.fn(),
  transcribeAudio: vi.fn()
}))

vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestGatewayForAgent: vi.fn()
}))

describe('usePromptActions sleep/wake session recovery', () => {
  const STORED_SESSION_ID = 'stored-db-xyz789'
  const RECOVERED_SESSION_ID = 'rt-recovered-456'

  afterEach(() => {
    cleanup()
    $turnStartedAt.set(null)
    vi.restoreAllMocks()
  })

  it('resumes the stored session and retries once when prompt.submit reports "session not found"', async () => {
    // After sleep/wake the gateway's in-memory session table is cleared, so the
    // first prompt.submit with the stale runtime id fails. The hook resumes the
    // durable stored id (which survives gateway restarts), gets a fresh live id,
    // and retries the send transparently.
    const calls: { method: string; params?: Record<string, unknown> }[] = []
    let submitAttempts = 0

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'prompt.submit') {
        submitAttempts += 1

        if (submitAttempts === 1) {
          throw new Error('session not found')
        }

        return {} as never
      }

      if (method === 'session.resume') {
        return { session_id: RECOVERED_SESSION_ID } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        storedSessionId={STORED_SESSION_ID}
      />
    )

    const ok = await handle!.submitText('message after wake')

    expect(ok).toBe(true)
    // First submit (stale id) → session.resume (stored id) → retry submit (fresh id).
    expect(calls.map(c => c.method)).toEqual(['prompt.submit', 'session.resume', 'prompt.submit'])
    expect(calls[1]?.params).toEqual({ session_id: STORED_SESSION_ID, source: 'desktop', omit_messages: true })
    expect(calls[2]?.params).toEqual({
      submission_id: expect.any(String),
      session_id: RECOVERED_SESSION_ID,
      text: 'message after wake'
    })
  })

  it('publishes the recovered runtime binding before retrying through the remote owner router', async () => {
    const calls: { method: string; params?: Record<string, unknown> }[] = []
    let bindingPublished = false
    let submitAttempts = 0

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'prompt.submit') {
        submitAttempts += 1

        if (submitAttempts === 1) {
          throw new JsonRpcGatewayError('session not found', { code: 4001 })
        }

        if (!bindingPublished) {
          throw new JsonRpcGatewayError('session not found on ambient gateway', { code: 4001 })
        }

        return {} as never
      }

      if (method === 'session.resume') {
        return { session_id: RECOVERED_SESSION_ID } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        onUpdateState={(runtimeId, storedId) => {
          if (runtimeId === RECOVERED_SESSION_ID && storedId === STORED_SESSION_ID) {
            bindingPublished = true
          }
        }}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        storedSessionId={STORED_SESSION_ID}
      />
    )

    expect(await handle!.submitText('remote follow-up after reap')).toBe(true)
    expect(bindingPublished).toBe(true)
    expect(calls.map(call => call.method)).toEqual(['prompt.submit', 'session.resume', 'prompt.submit'])
    expect(calls[2]?.params).toEqual({
      submission_id: expect.any(String),
      session_id: RECOVERED_SESSION_ID,
      text: 'remote follow-up after reap'
    })
  })

  it('resumes the stored session and retries once when reloadFromMessage (regenerate) reports "session not found"', async () => {
    // reloadFromMessage builds its own prompt.submit call inline instead of
    // going through the shared send() path submitText/redirectPrompt use, so
    // it needs the same sleep/wake recovery independently — otherwise
    // "Regenerate" on a stale session surfaces a raw error instead of
    // silently resuming, same as the general submit case above.
    //
    // reloadFromMessage bails early on $busy — an earlier suite in this file
    // can leave it true (see the stale-closure describe block's own note),
    // so reset it defensively rather than relying on run order.
    $busy.set(false)
    setMessages([
      { id: 'u1', parts: [textPart('original prompt')], role: 'user', timestamp: 0 },
      { id: 'a1', parts: [textPart('reply')], role: 'assistant', timestamp: 1 }
    ] as never)

    const calls: { method: string; params?: Record<string, unknown> }[] = []
    let submitAttempts = 0

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'prompt.submit') {
        submitAttempts += 1

        if (submitAttempts === 1) {
          throw new Error('session not found')
        }

        return {} as never
      }

      if (method === 'session.resume') {
        return { session_id: RECOVERED_SESSION_ID } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        storedSessionId={STORED_SESSION_ID}
      />
    )

    await handle!.reloadFromMessage('u1')

    // First submit (stale id) → session.resume (stored id) → retry submit (fresh id).
    expect(calls.map(c => c.method)).toEqual(['prompt.submit', 'session.resume', 'prompt.submit'])
    expect(calls[1]?.params).toEqual({ session_id: STORED_SESSION_ID, source: 'desktop', omit_messages: true })
    expect(calls[2]?.params).toEqual(
      expect.objectContaining({ session_id: RECOVERED_SESSION_ID, text: 'original prompt' })
    )
  })

  // #67603 (second symptom): a recovery resume must re-register on the session's
  // OWNING profile. Resuming on whichever profile is live forks the conversation
  // into the wrong profile's DB — the session then appears under both profiles.
  it('carries the owning profile from the cache into the recovery resume', async () => {
    setSessions(() => [sessionInfo({ id: STORED_SESSION_ID, profile: 'work' })])

    const calls: { method: string; params?: Record<string, unknown> }[] = []
    let submitAttempts = 0

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'prompt.submit') {
        submitAttempts += 1

        if (submitAttempts === 1) {
          throw new Error('session not found')
        }

        return {} as never
      }

      if (method === 'session.resume') {
        return { session_id: RECOVERED_SESSION_ID } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        storedSessionId={STORED_SESSION_ID}
      />
    )

    expect(await handle!.submitText('message after wake')).toBe(true)
    expect(calls[1]?.params).toEqual({
      session_id: STORED_SESSION_ID,
      source: 'desktop',
      omit_messages: true,
      profile: 'work'
    })

    setSessions(() => [])
  })

  // The session lives on another profile and is outside the paginated sidebar
  // cache: resolve it by id across profiles rather than resuming profile-blind.
  it('resolves the owning profile across profiles when the session is not cached', async () => {
    // module-factory vi.fn is not reset by restoreAllMocks — reset explicitly in
    // the finally below so this resolved value never leaks into sibling tests.
    setSessions(() => [])
    vi.mocked(getSession).mockResolvedValue(sessionInfo({ id: STORED_SESSION_ID, profile: 'work' }))

    const calls: { method: string; params?: Record<string, unknown> }[] = []
    let submitAttempts = 0

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'prompt.submit') {
        submitAttempts += 1

        if (submitAttempts === 1) {
          throw new Error('session not found')
        }

        return {} as never
      }

      if (method === 'session.resume') {
        return { session_id: RECOVERED_SESSION_ID } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        storedSessionId={STORED_SESSION_ID}
      />
    )

    expect(await handle!.submitText('message after wake')).toBe(true)
    expect(calls[1]?.params).toEqual({
      session_id: STORED_SESSION_ID,
      source: 'desktop',
      omit_messages: true,
      profile: 'work'
    })

    vi.mocked(getSession).mockReset()
    setSessions(() => [])
  })

  it('background queue resume uses the queued stored id and leaves foreground runtime selected', async () => {
    const calls: { method: string; params?: Record<string, unknown> }[] = []
    let submitAttempts = 0

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'prompt.submit') {
        submitAttempts += 1

        if (submitAttempts === 1) {
          throw new Error('session not found')
        }

        return {} as never
      }

      if (method === 'session.resume') {
        return { session_id: RECOVERED_SESSION_ID } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    render(
      <Harness
        // The central binding is stale in lockstep with the caller here: the
        // sleep/wake reaper only clears the GATEWAY's in-memory session, so
        // client-side state still swears by the old runtime id. That is what
        // routes this case to the reactive 404→resume→retry path instead of
        // the proactive binding check (covered by the cross-session drain
        // tests above).
        getRuntimeIdForStoredSession={storedId => (storedId === STORED_SESSION_ID ? 'rt-background-stale' : null)}
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        storedSessionId="stored-foreground"
      />
    )
    await waitFor(() => expect(handle).not.toBeNull())

    const ok = await handle!.submitText('queued background message after wake', {
      fromQueue: true,
      sessionId: 'rt-background-stale',
      storedSessionId: STORED_SESSION_ID
    })

    expect(ok).toBe(true)
    expect(calls.map(c => c.method)).toEqual(['prompt.submit', 'session.resume', 'prompt.submit'])
    expect(calls[0]?.params).toEqual({
      queued: true,
      session_id: 'rt-background-stale',
      submission_id: expect.any(String),
      text: 'queued background message after wake'
    })
    expect(calls[1]?.params).toEqual({
      session_id: STORED_SESSION_ID,
      source: 'desktop',
      omit_messages: true
    })
    expect(calls[2]?.params).toEqual({
      queued: true,
      session_id: RECOVERED_SESSION_ID,
      submission_id: expect.any(String),
      text: 'queued background message after wake'
    })
    expect(handle!.activeSessionIdRef.current).toBe(RUNTIME_SESSION_ID)
  })

  it('resumes the stored session and retries once when session.interrupt reports "session not found"', async () => {
    const calls: { method: string; params?: Record<string, unknown> }[] = []
    let interruptAttempts = 0

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'session.interrupt') {
        interruptAttempts += 1

        if (interruptAttempts === 1) {
          throw new Error('session not found')
        }

        return {} as never
      }

      if (method === 'session.resume') {
        return { session_id: RECOVERED_SESSION_ID } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        storedSessionId={STORED_SESSION_ID}
      />
    )
    await waitFor(() => expect(handle).not.toBeNull())

    await handle!.cancelRun()

    expect(calls.map(c => c.method)).toEqual(['session.interrupt', 'session.resume', 'session.interrupt'])
    expect(calls[0]?.params).toEqual({ session_id: RUNTIME_SESSION_ID })
    expect(calls[1]?.params).toEqual({
      session_id: STORED_SESSION_ID,
      source: 'desktop',
      omit_messages: true
    })
    expect(calls[2]?.params).toEqual({ session_id: RECOVERED_SESSION_ID })
  })

  it('clears the active and cached turn clocks when stopping a turn', async () => {
    const states: Record<string, unknown>[] = []
    const requestGateway = vi.fn(async () => ({}) as never)
    $turnStartedAt.set(1_700_000_000_000)

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        onSeedState={state => states.push(state)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
      />
    )

    await handle!.cancelRun()

    expect($turnStartedAt.get()).toBeNull()
    expect(states.at(-1)).toMatchObject({
      awaitingResponse: false,
      busy: false,
      interrupted: true,
      turnStartedAt: null
    })
  })

  it('surfaces the original error (no resume) when the failure is not "session not found"', async () => {
    const calls: string[] = []
    const states: Record<string, unknown>[] = []

    const requestGateway = vi.fn(async (method: string) => {
      calls.push(method)

      if (method === 'prompt.submit') {
        throw new Error('gateway exploded')
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        onSeedState={s => states.push(s)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        storedSessionId={STORED_SESSION_ID}
      />
    )

    // submitText swallows the error into an inline bubble and returns false.
    expect(await handle!.submitText('message')).toBe(false)
    // No resume attempt for a non-recoverable error.
    expect(calls).not.toContain('session.resume')
  })

  it('surfaces "session not found" (no resume) when there is no stored session id', async () => {
    const calls: string[] = []

    const requestGateway = vi.fn(async (method: string) => {
      calls.push(method)

      if (method === 'prompt.submit') {
        throw new Error('session not found')
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        storedSessionId={null}
      />
    )

    // With a null stored ref, the `&& selectedStoredSessionIdRef.current` guard
    // short-circuits — no resume is attempted and the error surfaces normally.
    expect(await handle!.submitText('message')).toBe(false)
    expect(calls).not.toContain('session.resume')
  })

  it('recovers via session.resume when prompt.submit TIMES OUT and a stored session is selected (#55578)', async () => {
    // A starved gateway loop rejects with "request timed out: prompt.submit".
    // With a stored session selected, that must recover exactly like
    // "session not found" — resume + retry — not surface an error that leaves
    // activeSessionId null and lets the next send mint a new session.
    const calls: { method: string; params?: Record<string, unknown> }[] = []
    let submitAttempts = 0

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'prompt.submit') {
        submitAttempts += 1

        if (submitAttempts === 1) {
          throw new Error('request timed out: prompt.submit')
        }

        return {} as never
      }

      if (method === 'session.resume') {
        return { session_id: RECOVERED_SESSION_ID } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        storedSessionId={STORED_SESSION_ID}
      />
    )

    const ok = await handle!.submitText('message during starved loop')

    expect(ok).toBe(true)
    expect(calls.map(c => c.method)).toEqual(['prompt.submit', 'session.resume', 'prompt.submit'])
    expect(calls[1]?.params).toEqual({
      session_id: STORED_SESSION_ID,
      source: 'desktop',
      omit_messages: true
    })
    expect(calls[2]?.params).toEqual({
      session_id: RECOVERED_SESSION_ID,
      submission_id: expect.any(String),
      text: 'message during starved loop'
    })
  })

  it('resumes the SELECTED stored session instead of minting a new one when activeSessionId is null (#55578 split)', async () => {
    // The exact split path from #55578 symptom (b): the runtime binding is
    // gone (orphan-reaped / cleared by a timeout) but a stored session is
    // still selected in the sidebar. A follow-up submit must continue that
    // conversation via session.resume — createBackendSessionForSend would
    // silently fork the user's chat in two.
    const calls: { method: string; params?: Record<string, unknown> }[] = []
    const createBackendSessionForSend = vi.fn(async () => 'brand-new-session-WRONG')

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'session.resume') {
        return { session_id: RECOVERED_SESSION_ID } as never
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        activeSessionId={null}
        createBackendSessionForSend={createBackendSessionForSend}
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        storedSessionId={STORED_SESSION_ID}
      />
    )

    const ok = await handle!.submitText('follow-up in the selected chat')

    expect(ok).toBe(true)
    expect(createBackendSessionForSend).not.toHaveBeenCalled()
    expect(calls.map(c => c.method)).toEqual(['session.resume', 'prompt.submit'])
    expect(calls[0]?.params).toEqual({
      session_id: STORED_SESSION_ID,
      source: 'desktop',
      omit_messages: true
    })
    expect(calls[1]?.params).toMatchObject({ session_id: RECOVERED_SESSION_ID })
  })

  it('never replaces a selected stored session when its direct runtime resume fails', async () => {
    const activeSessionIdRef: MutableRefObject<string | null> = { current: null }
    const busyRef: MutableRefObject<boolean> = { current: false }
    const createBackendSessionForSend = vi.fn(async () => 'brand-new-session-WRONG')

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.resume') {
        throw new Error('4007 session not found on the active profile')
      }

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        activeSessionId={null}
        activeSessionIdRef={activeSessionIdRef}
        busyRef={busyRef}
        createBackendSessionForSend={createBackendSessionForSend}
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        storedSessionId={STORED_SESSION_ID}
      />
    )

    expect(await handle!.submitText('keep me in the selected conversation')).toBe(false)
    expect(busyRef.current).toBe(false)
    expect(createBackendSessionForSend).not.toHaveBeenCalled()
    expect(requestGateway).not.toHaveBeenCalledWith('prompt.submit', expect.anything(), expect.anything())
  })

  it('resumes the ROUTED stored session instead of minting a new one when profile switching cleared both session refs', async () => {
    // A profile swap/reconnect can temporarily clear both volatile ids while
    // the durable route still points at the conversation the user is viewing.
    // Enter during that window must resume the routed chat, never create a
    // contextless session (or create it against the transient wrong profile).
    const activeSessionIdRef: MutableRefObject<string | null> = { current: 'rt-wrong-profile' }
    const selectedStoredSessionIdRef: MutableRefObject<string | null> = { current: null }
    let boundRuntimeId: string | null = null
    const createBackendSessionForSend = vi.fn(async () => 'brand-new-session-WRONG')
    const requestGateway = vi.fn(async () => ({}) as never)

    const resumeStoredSession = vi.fn(async (storedSessionId: string) => {
      expect(storedSessionId).toBe(STORED_SESSION_ID)
      selectedStoredSessionIdRef.current = STORED_SESSION_ID
      activeSessionIdRef.current = RECOVERED_SESSION_ID
      boundRuntimeId = RECOVERED_SESSION_ID
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        activeSessionId="rt-wrong-profile"
        activeSessionIdRef={activeSessionIdRef}
        createBackendSessionForSend={createBackendSessionForSend}
        getRoutedStoredSessionId={() => STORED_SESSION_ID}
        getRuntimeIdForStoredSession={() => boundRuntimeId}
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        resumeStoredSession={resumeStoredSession}
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
        storedSessionId={null}
      />
    )

    expect(await handle!.submitText('follow-up while the profile route is rebinding')).toBe(true)
    expect(resumeStoredSession).toHaveBeenCalledWith(STORED_SESSION_ID)
    expect(createBackendSessionForSend).not.toHaveBeenCalled()
    expect(requestGateway).toHaveBeenCalledWith(
      'prompt.submit',
      {
        submission_id: expect.any(String),
        session_id: RECOVERED_SESSION_ID,
        text: 'follow-up while the profile route is rebinding'
      },
      1_800_000
    )
  })

  it('lets the durable route replace a stale selected session and runtime before submit', async () => {
    const activeSessionIdRef: MutableRefObject<string | null> = { current: 'rt-wrong-profile' }
    const selectedStoredSessionIdRef: MutableRefObject<string | null> = { current: 'stored-wrong-profile' }
    let boundRuntimeId: string | null = null
    const requestGateway = vi.fn(async () => ({}) as never)

    const resumeStoredSession = vi.fn(async () => {
      selectedStoredSessionIdRef.current = STORED_SESSION_ID
      activeSessionIdRef.current = RECOVERED_SESSION_ID
      boundRuntimeId = RECOVERED_SESSION_ID
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        activeSessionId="rt-wrong-profile"
        activeSessionIdRef={activeSessionIdRef}
        getRoutedStoredSessionId={() => STORED_SESSION_ID}
        getRuntimeIdForStoredSession={() => boundRuntimeId}
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        resumeStoredSession={resumeStoredSession}
        runtimeIdByStoredSessionIdRef={{ current: new Map([[STORED_SESSION_ID, RECOVERED_SESSION_ID]]) }}
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
        storedSessionId={STORED_SESSION_ID}
      />
    )

    expect(await handle!.submitText('stay in the routed profile session')).toBe(true)
    expect(resumeStoredSession).toHaveBeenCalledWith(STORED_SESSION_ID)
    expect(requestGateway).toHaveBeenCalledWith(
      'prompt.submit',
      {
        submission_id: expect.any(String),
        session_id: RECOVERED_SESSION_ID,
        text: 'stay in the routed profile session'
      },
      1_800_000
    )
  })

  it('submits directly when the routed stored session already owns the live runtime', async () => {
    const activeSessionIdRef: MutableRefObject<string | null> = { current: RECOVERED_SESSION_ID }
    const selectedStoredSessionIdRef: MutableRefObject<string | null> = { current: STORED_SESSION_ID }
    const requestGateway = vi.fn(async () => ({}) as never)
    const resumeStoredSession = vi.fn()

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        activeSessionId={RECOVERED_SESSION_ID}
        activeSessionIdRef={activeSessionIdRef}
        getRoutedStoredSessionId={() => STORED_SESSION_ID}
        getRuntimeIdForStoredSession={() => RECOVERED_SESSION_ID}
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        resumeStoredSession={resumeStoredSession}
        runtimeIdByStoredSessionIdRef={{ current: new Map([[STORED_SESSION_ID, RECOVERED_SESSION_ID]]) }}
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
        storedSessionId={STORED_SESSION_ID}
      />
    )

    expect(await handle!.submitText('normal follow-up')).toBe(true)
    expect(resumeStoredSession).not.toHaveBeenCalled()
    expect(requestGateway).toHaveBeenCalledWith(
      'prompt.submit',
      { submission_id: expect.any(String), session_id: RECOVERED_SESSION_ID, text: 'normal follow-up' },
      1_800_000
    )
  })

  it('never falls through to session.create or a stale runtime when routed-session recovery fails', async () => {
    const activeSessionIdRef: MutableRefObject<string | null> = { current: 'rt-wrong-profile' }
    const selectedStoredSessionIdRef: MutableRefObject<string | null> = { current: STORED_SESSION_ID }
    const busyRef: MutableRefObject<boolean> = { current: false }
    let recoverySucceeds = false
    let boundRuntimeId: string | null = null

    const createBackendSessionForSend = vi.fn(async () => 'brand-new-session-WRONG')
    const requestGateway = vi.fn(async () => ({}) as never)

    const resumeStoredSession = vi.fn(async () => {
      if (!recoverySucceeds) {
        return
      }

      activeSessionIdRef.current = RECOVERED_SESSION_ID
      boundRuntimeId = RECOVERED_SESSION_ID
    })

    $messages.set([])

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        activeSessionId="rt-wrong-profile"
        activeSessionIdRef={activeSessionIdRef}
        busyRef={busyRef}
        createBackendSessionForSend={createBackendSessionForSend}
        getRoutedStoredSessionId={() => STORED_SESSION_ID}
        getRuntimeIdForStoredSession={() => boundRuntimeId}
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        resumeStoredSession={resumeStoredSession}
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
        storedSessionId={STORED_SESSION_ID}
      />
    )

    expect(await handle!.submitText('do not fork me')).toBe(false)
    expect(busyRef.current).toBe(false)
    expect($messages.get()).toEqual([])
    expect(resumeStoredSession).toHaveBeenCalledWith(STORED_SESSION_ID)
    expect(createBackendSessionForSend).not.toHaveBeenCalled()
    expect(requestGateway).not.toHaveBeenCalledWith('prompt.submit', expect.anything(), expect.anything())

    // Prove the failed attempt released the per-session submit lock. The next
    // send can recover and submit instead of being silently rejected forever.
    recoverySucceeds = true
    expect(await handle!.submitText('retry after recovery')).toBe(true)
    expect(requestGateway).toHaveBeenCalledWith(
      'prompt.submit',
      { submission_id: expect.any(String), session_id: RECOVERED_SESSION_ID, text: 'retry after recovery' },
      1_800_000
    )
  })

  it('still creates a new session for a genuine new-chat draft (no stored session selected)', async () => {
    const activeSessionIdRef: MutableRefObject<string | null> = { current: null }

    // Mirror the real createBackendSessionForSend: a successful create
    // re-homes the active runtime ref to the session it minted BEFORE
    // returning. An inert stub here is what let the new-chat drift-abort
    // regression ship green.
    const createBackendSessionForSend = vi.fn(async () => {
      activeSessionIdRef.current = RUNTIME_SESSION_ID

      return RUNTIME_SESSION_ID
    })

    const calls: string[] = []

    const requestGateway = vi.fn(async (method: string) => {
      calls.push(method)

      return {} as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        activeSessionId={null}
        activeSessionIdRef={activeSessionIdRef}
        createBackendSessionForSend={createBackendSessionForSend}
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        storedSessionId={null}
      />
    )

    const ok = await handle!.submitText('first message of a new chat')

    expect(ok).toBe(true)
    expect(createBackendSessionForSend).toHaveBeenCalledTimes(1)
    expect(calls).not.toContain('session.resume')
  })
})

import { JsonRpcGatewayError } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { type ChatMessage, chatMessageText, textPart } from '@/lib/chat-messages'
import { $composerAttachments } from '@/store/composer'
import { clearNotifications } from '@/store/notifications'
import { $awaitingResponse, $busy, setSessions } from '@/store/session'
import { dropSessionState } from '@/store/session-states'

import { clearSingleFlightSessionResumeState } from './single-flight-resume'
import { actRender, Harness, type HarnessHandle, RUNTIME_SESSION_ID, sessionInfo } from './test-harness'

vi.mock('@/hermes', () => ({
  getLatestSessionMessages: vi.fn(async () => ({ messages: [], session_id: 'session' })),
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  getSession: vi.fn(),
  PROMPT_SUBMIT_REQUEST_TIMEOUT_MS: 1_800_000,
  setApiRequestProfile: vi.fn(),
  transcribeAudio: vi.fn()
}))

const STORED_ID = 'stored-prompt-retry'
const RECOVERED_ID = 'runtime-prompt-retry'

function legacyRefusal() {
  return new JsonRpcGatewayError(
    'invalid params for prompt.submit: client_request_id: Extra inputs are not permitted',
    { code: 4000 }
  )
}

describe('prompt submit recovery', () => {
  beforeEach(() => {
    clearSingleFlightSessionResumeState()
    setSessions(() => [sessionInfo({ id: STORED_ID })])
    $composerAttachments.set([])
    $busy.set(false)
    $awaitingResponse.set(false)
    clearNotifications()
  })

  afterEach(() => {
    cleanup()
    dropSessionState(RUNTIME_SESSION_ID)
    dropSessionState(RECOVERED_ID)
    vi.restoreAllMocks()
  })

  it('never replays a legacy submit after its accepted acknowledgement times out', async () => {
    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'prompt.submit') {
        if (params?.client_request_id) {
          throw legacyRefusal()
        }

        throw new Error('request timed out: prompt.submit')
      }

      return { session_id: RECOVERED_ID } as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        storedSessionId={STORED_ID}
      />
    )

    expect(await handle!.submitText('accepted legacy prompt')).toBe(false)
    expect(requestGateway.mock.calls.map(call => call[0])).toEqual(['prompt.submit', 'prompt.submit'])
    expect(requestGateway.mock.calls[1][1]).toEqual({ session_id: RUNTIME_SESSION_ID, text: 'accepted legacy prompt' })
  })

  it('still recovers an explicit legacy session-not-found refusal', async () => {
    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'prompt.submit') {
        if (params?.client_request_id) {
          throw legacyRefusal()
        }

        if (params?.session_id === RUNTIME_SESSION_ID) {
          throw new Error('session not found')
        }

        return { status: 'streaming' } as never
      }

      return { session_id: RECOVERED_ID } as never
    })

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        storedSessionId={STORED_ID}
      />
    )

    expect(await handle!.submitText('refused legacy prompt')).toBe(true)
    expect(requestGateway.mock.calls.map(call => call[0])).toEqual([
      'prompt.submit', 'prompt.submit', 'session.resume', 'prompt.submit', 'prompt.submit'
    ])
    expect(requestGateway.mock.calls[4][1]).toEqual({ session_id: RECOVERED_ID, text: 'refused legacy prompt' })
  })

  it.each([false, true])('hydrates and settles a completed duplicate in the recovered runtime (background=%s)', async background => {
    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'prompt.submit') {
        if (params?.session_id === RUNTIME_SESSION_ID) {
          throw new Error('request timed out: prompt.submit')
        }

        return {
          duplicate: true,
          status: 'complete',
          messages: [
            { role: 'user', content: 'already finished', row_id: 41 },
            { role: 'assistant', content: 'recovered answer', row_id: 42 }
          ]
        } as never
      }

      return { session_id: RECOVERED_ID } as never
    })

    const busyRef = { current: background }

    $busy.set(background)
    $awaitingResponse.set(background)
    const updates: { sessionId: string; state: Record<string, unknown>; storedSessionId: null | string | undefined }[] = []
    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        busyRef={busyRef}
        getRuntimeIdForStoredSession={storedId => storedId === STORED_ID ? RUNTIME_SESSION_ID : null}
        onReady={h => (handle = h)}
        onUpdateState={(sessionId, storedSessionId, state) => updates.push({ sessionId, state, storedSessionId })}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        seedStreamId="missed-completion-stream"
        seedTurnStartedAt={123}
        storedSessionId={background ? 'foreground-stored' : STORED_ID}
      />
    )

    expect(await handle!.submitText('already finished', background ? {
      fromQueue: true, sessionId: RUNTIME_SESSION_ID, storedSessionId: STORED_ID
    } : undefined)).toBe(true)
    const settled = updates.at(-1)!

    expect(settled.sessionId).toBe(RECOVERED_ID)
    expect(settled.storedSessionId).toBe(STORED_ID)
    expect(settled.state).toMatchObject({
      busy: false, awaitingResponse: false, streamId: null, turnStartedAt: null, turnLive: false
    })
    const messages = settled.state.messages as ChatMessage[]

    expect(messages.map(chatMessageText)).toEqual(['already finished', 'recovered answer'])
    expect(messages.map(message => message.rowId)).toEqual([41, 42])
    expect(busyRef.current).toBe(background)
    expect($busy.get()).toBe(background)
    expect($awaitingResponse.get()).toBe(background)
    const submitCalls = requestGateway.mock.calls.filter(call => call[0] === 'prompt.submit')

    expect(submitCalls).toHaveLength(2)
    expect(submitCalls[1][1]?.client_request_id).toBe(submitCalls[0][1]?.client_request_id)
  })

  it.each(['new-start', 'same-stream', 'newer-settled'])('preserves newer state when a complete duplicate reply is delayed (%s)', async scenario => {
    let releaseReply: (reply: never) => void
    const delayedReply = new Promise<never>(resolve => { releaseReply = resolve })
    let reportDispatch: () => void
    const dispatched = new Promise<void>(resolve => { reportDispatch = resolve })
    let handle: HarnessHandle | null = null

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'prompt.submit') {
        if (params?.session_id === RUNTIME_SESSION_ID) {
          throw new Error('request timed out: prompt.submit')
        }

        reportDispatch!()

        return delayedReply
      }

      if (scenario === 'same-stream') {
        // A missed terminal event can leave the prior stream identity live.
        // The next message.start keeps that identity and the same clock.
        handle!.updateSessionState(RECOVERED_ID, state => ({
          ...state, turnLive: true, streamId: 'retained-stream', turnStartedAt: 123
        }))
      }

      return { session_id: RECOVERED_ID } as never
    })

    const busyRef = { current: false }
    let latest: Record<string, unknown> = {}
    await actRender(
      <Harness
        busyRef={busyRef}
        onReady={h => (handle = h)}
        onSeedState={state => { latest = state }}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        storedSessionId={STORED_ID}
      />
    )

    let pending: Promise<boolean>
    await act(async () => {
      pending = handle!.submitTextRaw('older intent')
      await dispatched
    })
    const live = scenario !== 'newer-settled'

    const newerMessages: ChatMessage[] = [{
      id: 'newer-turn', role: live ? 'assistant' : 'user', parts: [textPart('newer turn content')],
      ...(live ? {} : { rowId: 43 })
    }]

    await act(async () => {
      handle!.updateSessionState(RECOVERED_ID, state => ({
        ...state,
        messages: newerMessages,
        busy: live,
        awaitingResponse: live,
        turnLive: live,
        // message.start arrives before any delta allocates a stream bubble.
        streamId: scenario === 'same-stream' ? state.streamId : null,
        turnStartedAt: scenario === 'same-stream' ? state.turnStartedAt : live ? 456 : null
      }))
    })
    await act(async () => {
      releaseReply!({
        duplicate: true, status: 'complete', messages: [
          { role: 'user', text: 'older intent', row_id: 41 },
          { role: 'assistant', text: 'older answer', row_id: 42 }
        ]
      } as never)
      expect(await pending!).toBe(true)
    })

    expect(latest.messages).toBe(newerMessages)
    expect(latest).toMatchObject({ busy: live, awaitingResponse: live, turnLive: live })

    if (live) {
      expect(busyRef.current).toBe(true)
      expect($busy.get()).toBe(true)
      expect($awaitingResponse.get()).toBe(true)
    }

    expect(requestGateway.mock.calls.filter(call => call[0] === 'prompt.submit')).toHaveLength(2)
  })
})

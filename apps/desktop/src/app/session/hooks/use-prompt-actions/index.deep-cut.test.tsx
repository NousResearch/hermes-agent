import { JsonRpcGatewayError } from '@hermes/shared'
import { cleanup } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { textPart } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $confirmRequest, settleConfirm } from '@/store/confirm'
import { clearNotifications } from '@/store/notifications'
import { $busy, setMessages } from '@/store/session'
import { dropSessionState, publishSessionState } from '@/store/session-states'

import { actRender, Harness, type HarnessHandle, RUNTIME_SESSION_ID } from './prompt-actions-harness'

// Deep-cut confirms (#133716): regenerate/edit that would archive later user turns.

vi.mock('@/hermes', () => ({
  getLatestSessionMessages: vi.fn(async () => ({ messages: [], session_id: 'session' })),
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  getSession: vi.fn(),
  PROMPT_SUBMIT_REQUEST_TIMEOUT_MS: 1_800_000,
  setApiRequestProfile: vi.fn(),
  transcribeAudio: vi.fn()
}))

const RUNTIME_SESSION_B = 'rt-session-b-current'

type GatewayMock = Parameters<typeof Harness>[0]['requestGateway'] & { mock: { calls: unknown[][] } }

describe('usePromptActions deep-cut confirm across a session switch (#133716)', () => {
  afterEach(() => {
    cleanup()
    clearNotifications()
    setMessages([])
    $busy.set(false)
  })

  it('drops a 4033 confirm answered after the user switched sessions (#133716)', async () => {
    const submits: Record<string, unknown>[] = []

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'prompt.submit') {
        submits.push(params ?? {})

        if (!params?.confirm_deep_truncate) {
          throw new JsonRpcGatewayError('truncation would archive later user turns', { code: 4033 })
        }
      }

      return {} as never
    }) as unknown as GatewayMock

    const activeSessionIdRef: MutableRefObject<string | null> = { current: RUNTIME_SESSION_ID }
    setMessages([
      { id: 'u1', parts: [textPart('prompt A')], role: 'user', timestamp: 0 },
      { id: 'a1', parts: [textPart('reply')], role: 'assistant', timestamp: 1 }
    ] as never)

    let handle: HarnessHandle | null = null
    await actRender(
      <Harness
        activeSessionId={RUNTIME_SESSION_ID}
        activeSessionIdRef={activeSessionIdRef}
        onReady={h => (handle = h)}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        selectedStoredSessionIdRef={{ current: null }}
      />
    )

    // Session A's tail regenerate is refused; the user switches to B, then accepts.
    const stopConfirming = $confirmRequest.listen(request => {
      if (request) {
        activeSessionIdRef.current = RUNTIME_SESSION_B
        settleConfirm(true)
      }
    })

    await handle!.reloadFromMessage(null)
    stopConfirming()

    expect(submits).toHaveLength(1)
    expect(submits[0]).toMatchObject({ session_id: RUNTIME_SESSION_ID })
  })
})

describe('usePromptActions deep edit confirmed while output streams (#133716)', () => {
  afterEach(() => {
    cleanup()
    clearNotifications()
    setMessages([])
    $busy.set(false)
  })

  it('a confirmed deep edit whose turn gained a later user turn during the dialog sends nothing', async () => {
    const seed = [
      { id: 'u1', parts: [textPart('first')], role: 'user' as const, rowId: 11, timestamp: 0 },
      { id: 'a1', parts: [textPart('reply')], role: 'assistant' as const, timestamp: 1 },
      { id: 'u2', parts: [textPart('later')], role: 'user' as const, rowId: 13, timestamp: 2 }
    ]

    setMessages(seed as never)
    const submits: Record<string, unknown>[] = []

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'prompt.submit') {
        submits.push(params ?? {})
      }

      return {} as never
    })

    let handle: HarnessHandle | undefined

    await actRender(
      <Harness
        onReady={h => {
          handle = h
        }}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        seedMessages={seed}
      />
    )

    const stopConfirming = $confirmRequest.listen(request => {
      if (request) {
        setMessages([
          ...seed,
          { id: 'u3', parts: [textPart('unseen')], role: 'user', rowId: 15, timestamp: 3 }
        ] as never)
        settleConfirm(true)
      }
    })

    await handle!.editMessage({
      content: [{ text: 'edited first', type: 'text' }],
      parentId: null,
      role: 'user',
      sourceId: 'u1'
    } as never)
    stopConfirming()

    expect(submits).toEqual([])
  })

  it('re-aims a confirmed deep edit at the grown transcript instead of dropping it', async () => {
    const seed = [
      { id: 'u1', parts: [textPart('first')], role: 'user' as const, rowId: 11, timestamp: 0 },
      { id: 'a1', parts: [textPart('reply')], role: 'assistant' as const, timestamp: 1 },
      { id: 'u2', parts: [textPart('later')], role: 'user' as const, rowId: 13, timestamp: 2 },
      { id: 'a2', parts: [textPart('later reply')], role: 'assistant' as const, timestamp: 3 }
    ]

    setMessages(seed as never)
    const submits: Record<string, unknown>[] = []

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'prompt.submit') {
        submits.push(params ?? {})
      }

      return {} as never
    })

    let handle: HarnessHandle | undefined

    await actRender(
      <Harness
        onReady={h => {
          handle = h
        }}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        seedMessages={seed}
      />
    )

    // A stream delta replaces the transcript array while the user reads the dialog.
    const stopConfirming = $confirmRequest.listen(request => {
      if (request) {
        setMessages([...seed, { id: 'a2b', parts: [textPart('more')], role: 'assistant', timestamp: 4 }] as never)
        settleConfirm(true)
      }
    })

    await handle!.editMessage({
      content: [{ text: 'edited first', type: 'text' }],
      parentId: null,
      role: 'user',
      sourceId: 'u1'
    } as never)
    stopConfirming()

    expect(submits).toHaveLength(1)
    expect(submits[0]).toMatchObject({ confirm_deep_truncate: true, truncate_before_row_id: 11, text: 'edited first' })
  })
})

describe('usePromptActions deep-edit consent scope (#133716)', () => {
  afterEach(() => {
    cleanup()
    clearNotifications()
    setMessages([])
    $busy.set(false)
    dropSessionState(RUNTIME_SESSION_ID)
    dropSessionState('other-runtime')
  })

  it('a stale-target retry whose refresh shows a new later turn does not reuse the deep confirm', async () => {
    const seed = [
      { id: 'u1', parts: [textPart('first')], role: 'user', rowId: 11, timestamp: 0 },
      { id: 'a1', parts: [textPart('reply')], role: 'assistant', timestamp: 1 },
      { id: 'u2', parts: [textPart('second')], role: 'user', rowId: 13, timestamp: 2 }
    ]

    const fresh = [
      { ...seed[0], rowId: 21 },
      seed[1],
      { ...seed[2], rowId: 23 },
      { id: 'u3', parts: [textPart('arrived after consent')], role: 'user', rowId: 25, timestamp: 3 }
    ]

    setMessages(seed as never)
    const submits: Record<string, unknown>[] = []

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'prompt.submit') {
        submits.push(params ?? {})

        if (submits.length === 1) {
          throw new JsonRpcGatewayError('target user message is no longer in session history', { code: 4018 })
        }
      }

      return {} as never
    })

    let handle: HarnessHandle | undefined

    await actRender(
      <Harness
        onReady={h => {
          handle = h
        }}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        resumeStoredSession={async () => {
          setMessages(fresh as never)
        }}
        seedMessages={seed}
        storedSessionId="stored"
      />
    )

    const stopConfirming = $confirmRequest.listen(request => {
      if (request) {
        settleConfirm(true)
      }
    })

    await handle!.editMessage({
      content: [{ text: 'edited first', type: 'text' }],
      parentId: null,
      role: 'user',
      sourceId: 'u1'
    } as never)
    stopConfirming()

    expect(submits).toHaveLength(2)
    expect(submits[1]).toMatchObject({ truncate_before_row_id: 21 })
    expect(submits[1]).not.toHaveProperty('confirm_deep_truncate')
  })

  it('a deep-edit confirm answered after a session switch sends nothing and leaves the new session idle', async () => {
    const seed = [
      { id: 'u1', parts: [textPart('first')], role: 'user', rowId: 11, timestamp: 0 },
      { id: 'u2', parts: [textPart('second')], role: 'user', rowId: 13, timestamp: 1 }
    ]

    const other = [{ id: 'other-user', parts: [textPart('other chat')], role: 'user', rowId: 99, timestamp: 2 }]

    publishSessionState(RUNTIME_SESSION_ID, createClientSessionState('stored', seed as never))
    publishSessionState('other-runtime', createClientSessionState('other-stored', other as never))
    setMessages(seed as never)
    const activeSessionIdRef: MutableRefObject<string | null> = { current: RUNTIME_SESSION_ID }
    const submits: Record<string, unknown>[] = []

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'prompt.submit') {
        submits.push(params ?? {})
      }

      return {} as never
    })

    let handle: HarnessHandle | undefined

    await actRender(
      <Harness
        activeSessionIdRef={activeSessionIdRef}
        onReady={h => {
          handle = h
        }}
        refreshSessions={async () => undefined}
        requestGateway={requestGateway}
        seedMessages={seed}
      />
    )

    const stopConfirming = $confirmRequest.listen(request => {
      if (request) {
        activeSessionIdRef.current = 'other-runtime'
        setMessages(other as never)
        settleConfirm(true)
      }
    })

    await handle!.editMessage({
      content: [{ text: 'edited first', type: 'text' }],
      parentId: null,
      role: 'user',
      sourceId: 'u1'
    } as never)
    stopConfirming()

    expect(submits).toEqual([])
    expect($busy.get()).toBe(false)
  })
})

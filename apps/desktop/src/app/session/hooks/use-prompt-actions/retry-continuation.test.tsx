// Regression for #137277: the real error-card action must preserve completed work.
import { AssistantRuntimeProvider, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render, renderHook, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { Thread } from '@/components/assistant-ui/thread'
import { stubThreadEnvironment } from '@/components/assistant-ui/test-utils'
import type { ChatMessage } from '@/lib/chat-messages'
import { textPart } from '@/lib/chat-messages'
import { createClientSessionState, toRuntimeMessage } from '@/lib/chat-runtime'
import { $busy, setMessages } from '@/store/session'
import {
  $sessionStates,
  $sessionTiles,
  clearAllSessionStates,
  publishSessionState,
  setSessionTileDelegate
} from '@/store/session-states'
import { useSessionTileActions } from '@/app/chat/session-tile-actions'
import { MAIN_COMPOSER_SCOPE } from '@/app/chat/composer/scope'

import { actRender, Harness, type HarnessHandle } from './prompt-actions-harness'
import {
  adoptRecoveredRetry,
  applyRetryOptimistic,
  rollbackFailedRetry,
  planFailedRetry,
  promptAdmissionRejected,
  submitFailedRetry
} from './retry'
import { JsonRpcGatewayError } from '@hermes/shared'

vi.mock('@/hermes', () => ({
  getLatestSessionMessages: vi.fn(async () => ({ messages: [], session_id: 'session' })),
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  getSession: vi.fn(),
  PROMPT_SUBMIT_REQUEST_TIMEOUT_MS: 1_800_000,
  setApiRequestProfile: vi.fn(),
  transcribeAudio: vi.fn()
}))

stubThreadEnvironment()
const rows = [
  { id: 'u1', role: 'user', rowId: 11, parts: [textPart('do the work')], timestamp: 1 },
  {
    id: 'a1',
    role: 'assistant',
    interim: true,
    parts: [
      textPart('work completed'),
      {
        type: 'tool-call',
        toolCallId: 'call-1',
        toolName: 'terminal',
        args: { command: 'count' },
        argsText: '{"command":"count"}',
        result: { output: 'count=1' },
        completedAt: 2
      }
    ],
    timestamp: 2
  },
  {
    id: 'a-error',
    role: 'assistant',
    error: 'HTTP 429',
    retryToken: 'checkpoint-1',
    errorSurface: { layer: 'provider', code: 'rate_limit', retryable: true },
    parts: [],
    timestamp: 3
  }
] as ChatMessage[]

function RetryCard({ reload }: { reload: HarnessHandle['reloadFromMessage'] }) {
  const runtime = useExternalStoreRuntime({
    messages: rows.map(toRuntimeMessage),
    isRunning: false,
    onNew: async () => {},
    onReload: reload
  })
  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

afterEach(() => {
  cleanup()
  $sessionTiles.set([])
  clearAllSessionStates()
  setMessages([])
  $busy.set(false)
})

it('clicking Retry invokes continuation without truncating completed tool/interim rows', async () => {
  setMessages(rows)
  let handle: HarnessHandle | undefined
  const requestGateway = vi.fn(async () => ({}) as never)
  await actRender(
    <Harness
      onReady={h => {
        handle = h
      }}
      requestGateway={requestGateway}
      refreshSessions={async () => undefined}
      seedMessages={rows}
    />
  )
  render(<RetryCard reload={handle!.reloadFromMessageRaw} />)
  await act(async () => {
    fireEvent.click(await screen.findByRole('button', { name: 'Retry' }))
  })
  expect(requestGateway).toHaveBeenCalledWith(
    'prompt.retry',
    expect.objectContaining({ retry_token: 'checkpoint-1' }),
    expect.any(Number)
  )
  expect(requestGateway.mock.calls.some(call => (call as unknown[])[0] === 'prompt.submit')).toBe(false)
})

it('only resubmits an explicitly rejected admission; a lost acknowledgement never replays work', () => {
  const admission = [
    { ...rows[0], rowId: undefined },
    { ...rows[2], retryToken: undefined, admissionRejected: true }
  ]
  expect(planFailedRetry(admission, 'a-error')).toEqual({ text: 'do the work' })
  expect(() =>
    planFailedRetry([{ ...admission[0] }, { ...admission[1], admissionRejected: false }], 'a-error')
  ).toThrow(/checkpoint/)
  expect(promptAdmissionRejected(new JsonRpcGatewayError('retiring', { code: 5035 }))).toBe(true)
  expect(promptAdmissionRejected(new Error('response timed out'))).toBe(false)
  expect(() => planFailedRetry([{ ...rows[0] }, { ...admission[1] }], 'a-error')).toThrow(/checkpoint/)
})

it('drops a stale error-card intent after a newer user turn', async () => {
  const newer = [...rows, { id: 'u2', role: 'user' as const, parts: [textPart('new intent')] }]
  setMessages(newer)
  let handle: HarnessHandle | undefined
  const requestGateway = vi.fn(async () => ({}) as never)
  await actRender(
    <Harness
      onReady={h => {
        handle = h
      }}
      requestGateway={requestGateway}
      refreshSessions={async () => undefined}
      seedMessages={newer}
    />
  )
  await handle!.reloadFromMessage('u1', { runConfig: { custom: { retryMessageId: 'a-error' } } })
  expect(requestGateway).not.toHaveBeenCalled()
})

it('an explicit Regenerate still rewinds a durable failed user turn', async () => {
  const adjacent = [rows[0], rows[2]]
  setMessages(adjacent)
  let handle: HarnessHandle | undefined
  const requestGateway = vi.fn(async () => ({}) as never)
  await actRender(
    <Harness
      onReady={h => {
        handle = h
      }}
      requestGateway={requestGateway}
      refreshSessions={async () => undefined}
      seedMessages={adjacent}
    />
  )
  await handle!.reloadFromMessage('u1')
  expect(requestGateway).toHaveBeenCalledWith(
    'prompt.submit',
    expect.objectContaining({
      truncate_before_row_id: 11,
      confirm_truncate: true
    }),
    expect.any(Number)
  )
})

it('tiles use the same one-shot continuation and keep completed tool rows visible', async () => {
  const runtimeId = 'tile-retry'
  const storedId = 'stored-tile-retry'
  $sessionTiles.set([{ runtimeId, storedSessionId: storedId }])
  publishSessionState(runtimeId, createClientSessionState(storedId, rows))
  setSessionTileDelegate({
    updateSession: (_runtimeId, updater) => {
      const next = updater($sessionStates.get()[runtimeId])
      publishSessionState(runtimeId, next)
      return next
    }
  } as Parameters<typeof setSessionTileDelegate>[0])
  let resolveRequest: () => void = () => {}
  const requestGateway = vi.fn(async () => {
    await new Promise<void>(resolve => {
      resolveRequest = resolve
    })
    return {} as never
  })
  const { result } = renderHook(() =>
    useSessionTileActions({ requestGateway, runtimeId, storedSessionId: storedId, scope: MAIN_COMPOSER_SCOPE })
  )
  const config = { runConfig: { custom: { retryMessageId: 'a-error' } } }
  const first = result.current.reloadFromMessage('u1', config)
  await result.current.reloadFromMessage('u1', config)
  expect(requestGateway).toHaveBeenCalledTimes(1)
  expect(requestGateway).toHaveBeenCalledWith(
    'prompt.retry',
    expect.objectContaining({
      session_id: runtimeId,
      retry_token: 'checkpoint-1'
    }),
    expect.any(Number)
  )
  expect($sessionStates.get()[runtimeId].messages.find(row => row.id === 'a1')?.hidden).toBeFalsy()
  resolveRequest()
  await first
})

it('a stale runtime recovery forwards the same token and never falls back to truncate', async () => {
  const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
    if (method === 'prompt.retry' && params?.session_id === 'old-runtime') {
      throw new JsonRpcGatewayError('session not found', { code: 4001 })
    }
    if (method === 'session.resume') {
      return { session_id: 'new-runtime' } as never
    }
    return {} as never
  })
  const recovered = vi.fn()
  await submitFailedRetry('old-runtime', 'stored-retry', { retry_token: 'checkpoint-1' }, requestGateway, recovered)
  expect(recovered).toHaveBeenCalledWith('new-runtime')
  expect(requestGateway).toHaveBeenCalledWith(
    'prompt.retry',
    expect.objectContaining({
      session_id: 'new-runtime',
      retry_token: 'checkpoint-1'
    }),
    expect.any(Number)
  )
  expect(requestGateway.mock.calls.some(([method]) => method === 'prompt.submit')).toBe(false)
})

it('a session switch during recovery abandons Retry before submitting to the recovered runtime', async () => {
  let owned = true
  const requestGateway = vi.fn(async (method: string) => {
    if (method === 'session.resume') {
      owned = false
      return { session_id: 'new-runtime' } as never
    }
    throw new JsonRpcGatewayError('session not found', { code: 4001 })
  })
  await expect(
    submitFailedRetry(
      'old-runtime',
      'stored-switch',
      { retry_token: 'checkpoint-1' },
      requestGateway,
      vi.fn(),
      () => owned
    )
  ).rejects.toThrow(/changed/)
  expect(requestGateway.mock.calls.filter(([method]) => method === 'prompt.retry')).toHaveLength(1)
})

it('late failures and recovered-runtime adoption preserve newer streaming or user state', () => {
  const before = createClientSessionState('stored', rows)
  const optimistic = applyRetryOptimistic(before, 'a-error')
  const newer = {
    ...optimistic,
    messages: [
      ...optimistic.messages,
      {
        id: 'stream-new',
        role: 'assistant' as const,
        parts: [textPart('new output')],
        pending: true
      }
    ]
  }
  expect(rollbackFailedRetry(newer, rows, optimistic.messages)).toBe(newer)
  expect(adoptRecoveredRetry(newer, optimistic)).toBe(newer)
  expect(rollbackFailedRetry(optimistic, rows, optimistic.messages).messages).toBe(rows)
})

import { JsonRpcGatewayError } from '@hermes/shared'
import { cleanup, renderHook } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'
import { textPart } from '@/lib/chat-messages'
import { setActiveSessionId, setMessages } from '@/store/session'
import { clearAllSessionStates } from '@/store/session-states'
import { $sessionStates, $sessionTiles, publishSessionState, setSessionTileDelegate } from '@/store/session-states'
import { createClientSessionState } from '@/lib/chat-runtime'
import { useSessionTileActions } from '@/app/chat/session-tile-actions'
import { MAIN_COMPOSER_SCOPE } from '@/app/chat/composer/scope'
import type { ClientSessionState } from '@/app/types'

import { actRender, Harness, type HarnessHandle, RUNTIME_SESSION_ID } from './prompt-actions-harness'
import { applyRetryOptimistic, planFailedRetry, promptAdmissionRejected, rollbackFailedRetry } from './retry'

afterEach(() => {
  cleanup()
  clearAllSessionStates()
  $sessionTiles.set([])
  setMessages([])
  setActiveSessionId(null)
})

it.each([
  ['main', 'timeout'],
  ['tile', 'timeout'],
  ['main', 'unconfirmed'],
  ['tile', 'unconfirmed'],
  ['main', 'refused'],
  ['tile', 'refused']
])('%s keeps fresh Retry permission safe after %s', async (surface, outcome) => {
  const messages: ChatMessage[] = [
    { id: 'user', role: 'user', parts: [textPart('continue')] },
    {
      id: 'failed',
      role: 'assistant',
      parts: [],
      error: 'busy',
      admissionRejected: true,
      errorSurface: { layer: 'gateway', code: 'SESSION_OWNER_BUSY', retryable: true }
    }
  ]
  const requestGateway = vi.fn(async () => {
    if (outcome === 'timeout') throw new Error('response timed out')
    throw new JsonRpcGatewayError('Retry was not confirmed', {
      code: outcome === 'refused' ? 5035 : 4090,
      data: { reason: outcome === 'refused' ? 'runtime_draining' : 'SESSION_OWNER_UNCONFIRMED' }
    })
  })
  let reload: HarnessHandle['reloadFromMessage']
  if (surface === 'main') {
    setActiveSessionId(RUNTIME_SESSION_ID)
    setMessages(messages)
    let handle: HarnessHandle | undefined
    await actRender(
      <Harness
        onReady={h => {
          handle = h
        }}
        requestGateway={requestGateway}
        onUpdateState={(id, _stored, state) => publishSessionState(id, state as unknown as ClientSessionState)}
        refreshSessions={async () => undefined}
        seedMessages={messages}
      />
    )
    reload = handle!.reloadFromMessage
  } else {
    const runtimeId = 'tile-retry'
    $sessionTiles.set([{ runtimeId, storedSessionId: runtimeId }])
    publishSessionState(runtimeId, createClientSessionState(runtimeId, messages))
    setSessionTileDelegate({
      updateSession: (_id, update) => {
        const next = update($sessionStates.get()[runtimeId])
        publishSessionState(runtimeId, next)
        return next
      }
    } as Parameters<typeof setSessionTileDelegate>[0])
    const { result } = renderHook(() =>
      useSessionTileActions({
        requestGateway,
        runtimeId,
        storedSessionId: runtimeId,
        scope: MAIN_COMPOSER_SCOPE
      })
    )
    reload = result.current.reloadFromMessage
  }
  const config = { runConfig: { custom: { retryMessageId: 'failed' } } }
  await reload('user', config)
  await reload('user', config)
  expect(requestGateway).toHaveBeenCalledTimes(outcome === 'refused' ? 2 : 1)
})

it('revokes a cached fresh proof without overwriting newer output or a new checkpoint', () => {
  const rows: ChatMessage[] = [
    { id: 'user', role: 'user', parts: [textPart('continue')] },
    { id: 'failed', role: 'assistant', parts: [], error: 'busy', admissionRejected: true }
  ]
  const original = createClientSessionState('stored', rows)
  const optimistic = applyRetryOptimistic(original, 'failed')
  const output: ChatMessage = { id: 'new-output', role: 'assistant', parts: [textPart('newer output')] }
  const cached = { ...original, busy: true, messages: [...rows.map(row => ({ ...row })), output] }
  const revoked = rollbackFailedRetry(cached, rows, optimistic.messages, 'failed')
  expect(revoked.busy).toBe(true)
  expect(revoked.messages.at(-1)).toBe(output)
  expect(revoked.messages[1].admissionRejected).toBe(false)
  const checkpoint = { ...cached, messages: [rows[0], { ...rows[1], retryToken: 'new-checkpoint' }] }
  expect(rollbackFailedRetry(checkpoint, rows, optimistic.messages, 'failed')).toBe(checkpoint)
})

it.each(['cached', 'checkpoint', 'durable-user'])('does not move fresh Retry onto a %s runtime', async kind => {
  const rows: ChatMessage[] = [
    { id: 'user', role: 'user', parts: [textPart('continue')] },
    { id: 'failed', role: 'assistant', parts: [], error: 'busy', admissionRejected: true }
  ]
  setActiveSessionId(RUNTIME_SESSION_ID)
  setMessages(rows)
  $sessionTiles.set([{ runtimeId: 'cached-runtime', storedSessionId: 'stored' }])
  const cachedRows = rows.map(row => ({ ...row }))
  if (kind === 'checkpoint') cachedRows[1].retryToken = 'new-checkpoint'
  if (kind === 'durable-user') cachedRows[0].rowId = 11
  publishSessionState('cached-runtime', createClientSessionState('stored', cachedRows))
  const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
    if (method === 'session.resume') return { session_id: 'cached-runtime' } as never
    if (params?.session_id === RUNTIME_SESSION_ID) throw new JsonRpcGatewayError('session not found', { code: 4001 })
    throw new Error('response timed out')
  })
  let handle: HarnessHandle | undefined
  await actRender(
    <Harness
      onReady={h => {
        handle = h
      }}
      requestGateway={requestGateway}
      onUpdateState={(id, _stored, state) => publishSessionState(id, state as unknown as ClientSessionState)}
      refreshSessions={async () => undefined}
      seedMessages={rows}
    />
  )
  const config = { runConfig: { custom: { retryMessageId: 'failed' } } }
  await handle!.reloadFromMessage('user', config)
  expect(requestGateway).toHaveBeenCalledTimes(1)
  expect(requestGateway).not.toHaveBeenCalledWith('session.resume', expect.anything(), expect.anything())
  expect($sessionStates.get()['cached-runtime'].messages).toBe(cachedRows)
  const calls = requestGateway.mock.calls.length
  await handle!.reloadFromMessage('user', config)
  expect(requestGateway).toHaveBeenCalledTimes(calls)
})

it.each([
  [4090, 'SESSION_OWNER_UNCONFIRMED'],
  [4001, 'storage_unavailable']
])('does not classify an uncertain %s / %s as a rejected admission', (code, reason) => {
  expect(
    promptAdmissionRejected(new JsonRpcGatewayError('The send may have committed', { code, data: { reason } }))
  ).toBe(false)
})

it.each(['unconfirmed', 'image', 'image-part'])('refuses %s at the planner and error-card action', async kind => {
  const user: ChatMessage = {
    id: 'user',
    role: 'user',
    parts: [textPart('describe this')],
    ...(kind === 'image' && { attachmentRefs: ['@image:/tmp/stale-upload.jpg'] })
  }
  if (kind === 'image-part') {
    user.parts.push({ type: 'image', image: 'https://example.com/stale.jpg' })
  }
  const failed: ChatMessage = {
    id: 'failed',
    role: 'assistant',
    error: 'Owner did not accept this image or confirm this send',
    parts: [],
    admissionRejected: true,
    errorSurface: {
      layer: 'gateway',
      code: kind === 'unconfirmed' ? 'SESSION_OWNER_UNCONFIRMED' : 'SESSION_OWNER_BUSY',
      retryable: kind !== 'unconfirmed'
    }
  }
  const messages = [user, failed]
  expect(() => planFailedRetry(messages, failed.id)).toThrow()
  setMessages(messages)
  let handle: HarnessHandle | undefined
  const requestGateway = vi.fn(async () => ({}) as never)
  await actRender(
    <Harness
      onReady={h => {
        handle = h
      }}
      requestGateway={requestGateway}
      refreshSessions={async () => undefined}
      seedMessages={messages}
    />
  )
  await handle!.reloadFromMessage(user.id, { runConfig: { custom: { retryMessageId: failed.id } } })
  expect(requestGateway).not.toHaveBeenCalled()
})

it('keeps definitive text-only BUSY resends and proven attachment continuations available', () => {
  const user: ChatMessage = { id: 'user', role: 'user', parts: [textPart('continue')] }
  const failed: ChatMessage = { id: 'failed', role: 'assistant', error: 'busy', parts: [], admissionRejected: true }
  expect(
    promptAdmissionRejected(
      new JsonRpcGatewayError('busy', {
        code: 4090,
        data: { reason: 'SESSION_OWNER_BUSY' }
      })
    )
  ).toBe(true)
  expect(planFailedRetry([user, failed], failed.id)).toEqual({ text: 'continue' })
  expect(
    planFailedRetry(
      [
        { ...user, attachmentRefs: ['@image:/tmp/proven.jpg'] },
        {
          ...failed,
          retryToken: 'durable-context'
        }
      ],
      failed.id
    )
  ).toEqual({ retry_token: 'durable-context' })
})

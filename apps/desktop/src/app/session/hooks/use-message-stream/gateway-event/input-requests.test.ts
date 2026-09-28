import { afterEach, describe, expect, it, vi } from 'vitest'

import { $sudoRequest, clearSudoRequest } from '@/store/prompts'
import { resetServerRequestsForTests } from '@/store/server-requests'
import { $activeSessionId } from '@/store/session'

import { handleInputRequestEvent } from './input-requests'
import { handleServerRequest } from './server-requests'
import type { ServerRequestContext } from './server-requests'
import type { GatewayEventContext } from './types'

vi.mock('@/store/native-notifications', () => ({ dispatchNativeNotification: vi.fn() }))

const deps = { updateSessionState: vi.fn(), upsertToolCall: vi.fn() } as unknown as ServerRequestContext['deps']

function cancel(requestId: string, routedSession: string | null): GatewayEventContext {
  const payload = { id: requestId, method: 'display.install.sudo', reason: 'timeout' }

  return {
    deps: deps as unknown as GatewayEventContext['deps'],
    event: { payload, type: 'request.cancel' },
    explicitSid: '',
    fromActiveSource: () => true,
    isActiveEvent: true,
    occurredAt: 1,
    payload: payload as GatewayEventContext['payload'],
    scheduleConfigRefresh: vi.fn(),
    sessionId: routedSession
  }
}

describe('Bot Screen install password card', () => {
  afterEach(() => {
    clearSudoRequest()
    resetServerRequestsForTests()
    $activeSessionId.set(null)
  })

  it('belongs to the app, not to the chat that happened to be open: it survives a chat switch and its cancel finds it', () => {
    // The gateway sends the request sessionless. Stored under the ambient chat, the card vanished the
    // moment the user switched chats and the cancel (routed to the NEW ambient chat) never reached it.
    $activeSessionId.set('chat-a')
    const respond = vi.fn()
    expect(
      handleServerRequest(
        {
          fail: vi.fn(),
          id: 'srq-1',
          method: 'display.install.sudo',
          params: { profile_key: '/home/h/.hermes', session_id: '' },
          profile: 'default',
          respond
        },
        deps,
        'chat-a'
      )
    ).toBe(true)
    expect($sudoRequest.get()?.requestId).toBe('srq-1')

    $activeSessionId.set('chat-b')
    expect($sudoRequest.get()?.requestId).toBe('srq-1')

    expect(handleInputRequestEvent(cancel('srq-1', 'chat-b'))).toBe(true)
    expect($sudoRequest.get()).toBeNull()
  })
})

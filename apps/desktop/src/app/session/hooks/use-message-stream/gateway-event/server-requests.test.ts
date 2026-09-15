import { afterEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { $clarifyRequests, clearClarifyRequest } from '@/store/clarify'
import { $toursEnabled } from '@/store/tours'

import { handleServerRequest } from './server-requests'
import type { ServerRequestContext } from './server-requests'

const deps = {
  activeSessionIdRef: { current: null },
  sessionInterrupted: () => false,
  updateSessionState: (_sessionId, update) => update(createClientSessionState('stored-session')),
  upsertToolCall: () => undefined
} as ServerRequestContext['deps']

function deliver(method: string, params: Record<string, unknown>, activeSessionId: null | string) {
  const respond = vi.fn()
  const fail = vi.fn()
  const handled = handleServerRequest({ fail, id: 'srq-1', method, params, profile: 'default', respond }, deps, activeSessionId)

  return { fail, handled, respond }
}

describe('connection request routing', () => {
  it('does not route connection operations through the server-request rail', () => {
    const { handled, respond } = deliver(
      'connection',
      {
        deadline_at: 1_800_000_000,
        op_id: 'op-1',
        session_id: 'session-a',
        targets: [{ action: 'install', kind: 'mcp', name: 'linear' }],
        timeout_seconds: 60,
        tool_call_id: 'call-1'
      },
      'session-a'
    )

    expect(handled).toBe(false)
    expect(respond).not.toHaveBeenCalled()
  })
})

describe('clarify request routing', () => {
  afterEach(() => {
    clearClarifyRequest()
    deps.activeSessionIdRef.current = null
  })

  it.each([
    ['a single question', { choices: ['yes', 'no'], question: 'Ship it?' }],
    [
      'a batch of questions',
      {
        questions: [
          { qid: 'q1', question: 'Choose a color' },
          { choices: ['small', 'large'], qid: 'q2', question: 'Choose a size' }
        ]
      }
    ]
  ])('stores %s under the runtime session id', (_kind, payload) => {
    deps.activeSessionIdRef.current = 'runtime-session'

    deliver('clarify', { ...payload, session_id: 'agent-session' }, 'runtime-session')

    expect($clarifyRequests.get()['runtime-session']).toMatchObject({ requestId: 'srq-1' })
    expect($clarifyRequests.get()['agent-session']).toBeUndefined()
  })
})

describe('preview action request routing', () => {
  it('leaves a scoped action request unanswered in a window showing another session', () => {
    const { handled, respond, fail } = deliver('preview.act', { action: 'elements', session_id: 'session-a' }, 'session-b')

    expect(handled).toBe(true)
    expect(respond).not.toHaveBeenCalled()
    expect(fail).not.toHaveBeenCalled()
  })

  it('fails fast for an unscoped request with no session in view', () => {
    const { respond } = deliver('preview.act', { action: 'elements' }, null)

    expect(respond).toHaveBeenCalledWith({
      value: JSON.stringify({
        error: 'The in-app browser only takes actions in the session the user is looking at.',
        success: false
      })
    })
  })
})

describe('tour request routing', () => {
  afterEach(() => {
    $toursEnabled.set(true)
  })

  it('leaves a scoped request unanswered in another session even when tours are disabled', () => {
    $toursEnabled.set(false)
    const { handled, respond } = deliver('tour', { action: 'discover', session_id: 'session-a' }, 'session-b')

    expect(handled).toBe(true)
    expect(respond).not.toHaveBeenCalled()
  })

  it('fails fast for an unscoped request with no session in view', () => {
    const { respond } = deliver('tour', { action: 'discover' }, null)

    expect(respond).toHaveBeenCalledWith({
      value: JSON.stringify({ error: 'Tours only run in the session the user is looking at.', success: false })
    })
  })
})

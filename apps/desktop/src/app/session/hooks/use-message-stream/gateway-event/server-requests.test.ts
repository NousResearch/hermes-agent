import { afterEach, describe, expect, it, vi } from 'vitest'

import { $sudoRequest, clearAllPrompts } from '@/store/prompts'
import { resetServerRequestsForTests } from '@/store/server-requests'
import { $activeSessionId } from '@/store/session'
import { $toursEnabled } from '@/store/tours'

import { handleServerRequest } from './server-requests'
import type { ServerRequestContext } from './server-requests'

const deps = {} as ServerRequestContext['deps']

function deliver(method: string, params: Record<string, unknown>, activeSessionId: null | string) {
  const respond = vi.fn()
  const fail = vi.fn()
  const handled = handleServerRequest({ fail, id: 'srq-1', method, params, profile: 'default', respond }, deps, activeSessionId)

  return { fail, handled, respond }
}

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

describe('sudo request routing', () => {
  afterEach(() => {
    clearAllPrompts()
    resetServerRequestsForTests()
    $activeSessionId.set(null)
  })

  it('parks the sudo request with the command being authorized', () => {
    $activeSessionId.set('s1')
    const respond = vi.fn()
    const fail = vi.fn()
    const updateSessionState = vi.fn()

    const handled = handleServerRequest(
      {
        fail,
        id: 'sudo-1',
        method: 'sudo',
        params: { command: 'sudo whoami', session_id: 's1' },
        profile: 'default',
        replayed: true,
        respond
      } as never,
      { sessionInterrupted: () => false, updateSessionState } as never,
      's1'
    )

    expect(handled).toBe(true)
    expect($sudoRequest.get()).toEqual({
      command: 'sudo whoami',
      requestId: 'sudo-1',
      sessionId: 's1'
    })
    expect(updateSessionState).toHaveBeenCalledWith('s1', expect.any(Function))
  })
})

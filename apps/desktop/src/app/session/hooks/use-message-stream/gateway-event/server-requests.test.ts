import type { PreviewActParams, TourParams } from '@hermes/shared'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import type { ScopedServerRequest } from '@/store/gateway'
import { $toursEnabled } from '@/store/tours'

import { handleServerRequest } from './server-requests'
import type { ServerRequestContext } from './server-requests'

const deps = {
  activeSessionIdRef: { current: null },
  sessionInterrupted: () => false,
  updateSessionState: (_sessionId, update) => update(createClientSessionState('stored-session')),
  upsertToolCall: () => undefined
} as ServerRequestContext['deps']

const previewAct = (session_id: string): PreviewActParams => ({
  action: 'elements',
  amount: null,
  full: null,
  key: null,
  max: null,
  ref: null,
  selector: null,
  session_id,
  submit: null,
  text: null,
  to: null
})

const tourParams = (session_id: string): TourParams => ({
  action: 'targets',
  selector: null,
  session_id,
  side: null,
  step_index: null,
  steps: null,
  surface: null,
  text: null,
  title: null
})

function deliver<M extends 'preview.act' | 'tour'>(
  method: M,
  params: ScopedServerRequest<M>['params'],
  activeSessionId: null | string
) {
  const respond = vi.fn()
  const fail = vi.fn()

  const request: ScopedServerRequest<M> = {
    fail,
    id: 'srq-1',
    method,
    params,
    profile: 'default',
    respond,
    sessionId: params.session_id || null
  }

  handleServerRequest(request, deps, activeSessionId)

  return { fail, respond }
}

describe('preview action request routing', () => {
  it('leaves a scoped action request unanswered in a window showing another session', () => {
    const { respond, fail } = deliver('preview.act', previewAct('session-a'), 'session-b')

    expect(respond).not.toHaveBeenCalled()
    expect(fail).not.toHaveBeenCalled()
  })

  it('fails fast for an unscoped request with no session in view', () => {
    const { respond } = deliver('preview.act', previewAct(''), null)

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
    const { respond } = deliver('tour', tourParams('session-a'), 'session-b')

    expect(respond).not.toHaveBeenCalled()
  })

  it('fails fast for an unscoped request with no session in view', () => {
    const { respond } = deliver('tour', tourParams(''), null)

    expect(respond).toHaveBeenCalledWith({
      value: JSON.stringify({ error: 'Tours only run in the session the user is looking at.', success: false })
    })
  })
})

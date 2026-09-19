import type { PreviewActParams, ServerRequestMap, TourParams } from '@hermes/shared'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import type { ScopedServerRequest } from '@/store/gateway'
import { setActiveSessionId, setSessions } from '@/store/session'
import { $sessionTiles } from '@/store/session-states'
import { $toursEnabled } from '@/store/tours'
import { approvalParams } from '@/test/contract'
import type { SessionInfo } from '@/types/hermes'

import { handleServerRequest, previewSessionRoute } from './server-requests'
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

function deliver<M extends keyof ServerRequestMap>(
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

describe('approval request routing', () => {
  const notify = vi.fn().mockResolvedValue(true)
  const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }

  beforeEach(() => {
    notify.mockClear()
    desktopWindow.hermesDesktop = { notify } as unknown as Window['hermesDesktop']
    setSessions([{ id: 'session-a', title: 'Fix the flaky test' } as SessionInfo])
    setActiveSessionId('session-b')
  })

  afterEach(() => {
    delete desktopWindow.hermesDesktop
    setSessions([])
    setActiveSessionId(null)
  })

  it('titles the parked approval toast with the session it belongs to', () => {
    deliver('approval', approvalParams({ command: 'rm -rf /', description: 'dangerous', request_id: 'r1', session_id: 'session-a' }), 'session-b')

    expect(notify).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'approval', title: 'Approval needed — Fix the flaky test' })
    )
  })
})

describe('preview action request routing', () => {
  it('retries a replayed scoped request only while no session is bound yet', () => {
    expect(previewSessionRoute({ replayed: true, sessionId: 'session-a', activeSessionId: null })).toBe('retry')
    expect(previewSessionRoute({ replayed: true, sessionId: 'session-a', activeSessionId: 'session-a' })).toBe('run')
    expect(previewSessionRoute({ replayed: true, sessionId: 'session-a', activeSessionId: 'session-b' })).toBe('ignore')
    expect(previewSessionRoute({ replayed: true, sessionId: '', activeSessionId: null })).toBe('run')
  })

  it('leaves a scoped action request unanswered in a window showing another session', () => {
    const { respond, fail } = deliver('preview.act', previewAct('session-a'), 'session-b')

    expect(respond).not.toHaveBeenCalled()
    expect(fail).not.toHaveBeenCalled()
  })

  it('leaves scoped pane reads unanswered in a window showing another session', async () => {
    const reads = [
      deliver('preview.read', { count: null, session_id: 'session-a', start: null }, 'session-b'),
      deliver('terminal.read', { count: null, session_id: 'session-a', start: null }, 'session-b'),
      deliver('window.read', { session_id: 'session-a' }, 'session-b')
    ]

    await Promise.resolve()

    for (const { respond } of reads) {
      expect(respond).not.toHaveBeenCalled()
    }
  })

  it('answers pane reads for a session hosted in one of this window\'s tiles', async () => {
    // The tile session is not the active one, but this window hosts it: its
    // panes are here, so an 'ignore' would stall the tool until its deadline.
    $sessionTiles.set([{ runtimeId: 'session-a', storedSessionId: 'stored-a' } as never])

    try {
      const reads = [
        deliver('preview.read', { count: null, session_id: 'session-a', start: null }, 'session-b'),
        deliver('terminal.read', { count: null, session_id: 'session-a', start: null }, 'session-b'),
        deliver('window.read', { session_id: 'session-a' }, 'session-b')
      ]

      await new Promise(resolve => setTimeout(resolve, 0))

      for (const { respond } of reads) {
        expect(respond).toHaveBeenCalledTimes(1)
      }
    } finally {
      $sessionTiles.set([])
    }
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

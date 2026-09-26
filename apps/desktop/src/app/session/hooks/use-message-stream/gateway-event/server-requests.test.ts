import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { setActiveSessionId, setSessions } from '@/store/session'
import { $sessionTiles } from '@/store/session-states'
import { $toursEnabled } from '@/store/tours'
import type { SessionInfo } from '@/types/hermes'

import { handleServerRequest, previewSessionRoute, requestNamesActiveSession } from './server-requests'
import type { ServerRequestContext } from './server-requests'

vi.mock('@/lib/tour', () => ({ runTour: vi.fn(async () => ({ ok: true })) }))

const deps = {
  activeSessionIdRef: { current: null },
  sessionInterrupted: () => false,
  sessionStateByRuntimeIdRef: { current: new Map() },
  updateSessionState: (_sessionId, update) => update(createClientSessionState('stored-session')),
  upsertToolCall: () => undefined
} as ServerRequestContext['deps']

function deliver(method: string, params: Record<string, unknown>, activeSessionId: null | string) {
  const respond = vi.fn()
  const fail = vi.fn()

  const handled = handleServerRequest(
    { fail, id: 'srq-1', method, params, profile: 'default', respond },
    deps,
    activeSessionId
  )

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
    deliver(
      'approval',
      { command: 'rm -rf /', description: 'dangerous', request_id: 'r1', session_id: 'session-a' },
      'session-b'
    )

    expect(notify).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'approval', title: expect.stringContaining('Fix the flaky test') })
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
    const { handled, respond, fail } = deliver(
      'preview.act',
      { action: 'elements', session_id: 'session-a' },
      'session-b'
    )

    expect(handled).toBe(true)
    expect(respond).not.toHaveBeenCalled()
    expect(fail).not.toHaveBeenCalled()
  })

  it('leaves scoped pane reads unanswered in a window showing another session', async () => {
    const reads = ['preview.read', 'terminal.read', 'window.read'].map(method =>
      deliver(method, { session_id: 'session-a' }, 'session-b')
    )

    await Promise.resolve()

    for (const { handled, respond } of reads) {
      expect(handled).toBe(true)
      expect(respond).not.toHaveBeenCalled()
    }
  })

  it("answers pane reads for a session hosted in one of this window's tiles", async () => {
    // The tile session is not the active one, but this window hosts it: its
    // panes are here, so an 'ignore' would stall the tool until its deadline.
    $sessionTiles.set([{ runtimeId: 'session-a', storedSessionId: 'stored-a' } as never])

    try {
      const reads = ['preview.read', 'terminal.read', 'window.read'].map(method =>
        deliver(method, { session_id: 'session-a' }, 'session-b')
      )

      await new Promise(resolve => setTimeout(resolve, 0))

      for (const { handled, respond } of reads) {
        expect(handled).toBe(true)
        expect(respond).toHaveBeenCalledTimes(1)
      }
    } finally {
      $sessionTiles.set([])
    }
  })

  it('fails fast for an unscoped request with no session in view', () => {
    const { respond } = deliver('preview.act', { action: 'elements' }, null)

    expect(JSON.parse(respond.mock.calls[0][0].value)).toMatchObject({ success: false })
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

    expect(JSON.parse(respond.mock.calls[0][0].value)).toMatchObject({ success: false })
  })

  it('runs the tour for a compression-rotated session driving the conversation on screen', async () => {
    // #122062: auto-compression rotates the runtime session id (and the stored
    // tip) while the pane keeps the durable id it navigated to. The request is
    // stamped with the rotated runtime id — the gate must still see that it
    // names the conversation on screen instead of refusing it.
    setSessions([{ id: 'stored-tip', _lineage_root_id: 'stored-root', _lineage_ids: ['stored-root'] } as SessionInfo])
    deps.sessionStateByRuntimeIdRef.current.set('runtime-2', createClientSessionState('stored-tip'))

    try {
      const { handled, respond } = deliver('tour', { action: 'discover', session_id: 'runtime-2' }, 'stored-root')

      expect(handled).toBe(true)
      await vi.waitFor(() => expect(respond).toHaveBeenCalledTimes(1))
      expect(JSON.parse(respond.mock.calls[0][0].value)).toMatchObject({ ok: true })
    } finally {
      deps.sessionStateByRuntimeIdRef.current.clear()
      setSessions([])
    }
  })

  it('still refuses a scoped request naming another conversation', () => {
    // The window hosts the request's session as a tile (so the request is
    // routed here rather than left unanswered), but the pane shows a different
    // conversation — the gate must still refuse.
    setSessions([{ id: 'stored-tip', _lineage_root_id: 'stored-root' } as SessionInfo])
    deps.sessionStateByRuntimeIdRef.current.set('runtime-2', createClientSessionState('stored-tip'))
    $sessionTiles.set([{ runtimeId: 'runtime-2', storedSessionId: 'stored-tip' } as never])

    try {
      const { respond } = deliver('tour', { action: 'discover', session_id: 'runtime-2' }, 'other-root')

      expect(JSON.parse(respond.mock.calls[0][0].value)).toMatchObject({
        error: expect.stringContaining('the session the user is looking at')
      })
    } finally {
      $sessionTiles.set([])
      deps.sessionStateByRuntimeIdRef.current.clear()
      setSessions([])
    }
  })
})

describe('session identity matching (runtime vs stored ids)', () => {
  afterEach(() => {
    setSessions([])
  })

  it('matches a compression-rotated request id to the conversation the pane holds', () => {
    setSessions([{ id: 'stored-tip', _lineage_root_id: 'stored-root', _lineage_ids: ['stored-root'] } as SessionInfo])
    const bindings: Record<string, string> = { 'runtime-1': 'stored-root', 'runtime-2': 'stored-tip' }
    const storedIdForRuntimeId = (id: string) => bindings[id]

    // The pane holds the durable/lineage id the user navigated to.
    expect(
      requestNamesActiveSession({ activeSessionId: 'stored-root', sessionId: 'runtime-2', storedIdForRuntimeId })
    ).toBe(true)
    // The pane holds a stale runtime id while the request carries the rotated one.
    expect(
      requestNamesActiveSession({ activeSessionId: 'runtime-1', sessionId: 'runtime-2', storedIdForRuntimeId })
    ).toBe(true)
    // Plain equality still short-circuits.
    expect(requestNamesActiveSession({ activeSessionId: 'runtime-2', sessionId: 'runtime-2' })).toBe(true)
    // A foreign conversation is still not the active one.
    expect(
      requestNamesActiveSession({
        activeSessionId: 'stored-root',
        sessionId: 'other-runtime',
        storedIdForRuntimeId: (id: string) => (id === 'other-runtime' ? 'other-stored' : undefined)
      })
    ).toBe(false)
  })

  it('does not merge branch siblings that only share a lineage root', () => {
    setSessions([
      { id: 'branch-a', _lineage_root_id: 'shared-root' } as SessionInfo,
      { id: 'branch-b', _lineage_root_id: 'shared-root' } as SessionInfo
    ])

    expect(requestNamesActiveSession({ activeSessionId: 'branch-a', sessionId: 'branch-b' })).toBe(false)
    expect(requestNamesActiveSession({ activeSessionId: 'shared-root', sessionId: 'branch-b' })).toBe(true)
  })

  it('stays false when either side is unscoped', () => {
    expect(requestNamesActiveSession({ activeSessionId: null, sessionId: 'runtime-2' })).toBe(false)
    expect(requestNamesActiveSession({ activeSessionId: 'stored-root', sessionId: '' })).toBe(false)
  })
})

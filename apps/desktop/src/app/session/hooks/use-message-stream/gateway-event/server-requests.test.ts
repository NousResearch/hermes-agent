import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { setActiveSessionId, setSelectedStoredSessionId, setSessions } from '@/store/session'
import { $sessionStates, $sessionTiles } from '@/store/session-states'
import { $toursEnabled } from '@/store/tours'
import type { SessionInfo } from '@/types/hermes'

import { handleServerRequest, previewSessionRoute } from './server-requests'
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
    const reads = ['preview.read', 'terminal.read'].map(method =>
      deliver(method, { session_id: 'session-a' }, 'session-b')
    )

    await Promise.resolve()

    for (const { handled, respond } of reads) {
      expect(handled).toBe(true)
      expect(respond).not.toHaveBeenCalled()
    }
  })

  it('answers window.read empty instead of stalling when no window claims the session (#121609)', async () => {
    // window.read has no per-window pane: an unclaimed request used to wait
    // out the tool's full 30s deadline because silence was the "not mine"
    // signal. Pane-owned reads still wait for their owner (#113348).
    const { handled, respond } = deliver('window.read', { session_id: 'session-a' }, 'session-b')

    await Promise.resolve()

    expect(handled).toBe(true)
    expect(respond).toHaveBeenCalledWith({ value: '' })
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

describe('window.read claim tolerance (#121609)', () => {
  beforeEach(() => {
    setSessions([
      { id: 'stored-a', title: 'HUD conversation', _lineage_root_id: 'root-a' } as SessionInfo
    ])
  })

  afterEach(() => {
    setSessions([])
    setActiveSessionId(null)
    setSelectedStoredSessionId(null)
    $sessionStates.set({})
    $sessionTiles.set([])
  })

  it('claims when the window shows the conversation under its stored id while the backend asks about the runtime id', () => {
    // The HUD state: active is the pre-handoff runtime (or nothing), but this
    // window has the conversation selected and its runtime id lineage-maps.
    setSelectedStoredSessionId('stored-a')

    expect(previewSessionRoute({ activeSessionId: 'runtime-x', replayed: false, sessionId: 'root-a' })).toBe('run')
    expect(previewSessionRoute({ activeSessionId: null, replayed: false, sessionId: 'root-a' })).toBe('run')
    // Plain stored-id ask (no rotation): selected matches directly.
    expect(previewSessionRoute({ activeSessionId: 'runtime-x', replayed: false, sessionId: 'stored-a' })).toBe('run')
  })

  it('maps an unknown runtime id through the session-state cache to the shown conversation', () => {
    // A resume rebound the runtime without this window re-deriving lineage:
    // the state cache records which stored id the runtime id belongs to.
    setSelectedStoredSessionId('stored-a')
    $sessionStates.set({ 'runtime-rotated': createClientSessionState('stored-a') })

    expect(previewSessionRoute({ activeSessionId: 'runtime-rotated', replayed: false, sessionId: 'runtime-rotated' })).toBe('run')
  })

  it('claims for a tile whose stored session lineage-matches the asked id', () => {
    $sessionTiles.set([{ runtimeId: 'tile-runtime', storedSessionId: 'stored-a' } as never])

    expect(previewSessionRoute({ activeSessionId: 'session-b', replayed: false, sessionId: 'root-a' })).toBe('run')
    expect(previewSessionRoute({ activeSessionId: 'session-b', replayed: false, sessionId: 'stored-a' })).toBe('run')
  })

  it('never claims a conversation this window does not show', () => {
    setSelectedStoredSessionId('stored-a')

    expect(previewSessionRoute({ activeSessionId: 'session-b', replayed: false, sessionId: 'session-unrelated' })).toBe('ignore')
    expect(previewSessionRoute({ activeSessionId: 'session-b', replayed: false, sessionId: 'root-other' })).toBe('ignore')
  })

  it('claims nothing without a shown conversation — a background session stays unclaimed', () => {
    // No selection, no tiles: the tolerant branch must stay inert so a window
    // midsession cannot answer for a background conversation it never showed.
    expect(previewSessionRoute({ activeSessionId: 'session-b', replayed: false, sessionId: 'root-a' })).toBe('ignore')
  })

  it('lets a shown-conversation window answer a window.read end to end without a resume', async () => {
    // The filed repro: HUD mode / post-handoff main window — no runtime claim,
    // only the stored selection. The request now answers (empty here, because
    // the test window exposes no readWindowBelow bridge) instead of stalling.
    setSelectedStoredSessionId('stored-a')

    const { handled, respond } = deliver('window.read', { session_id: 'root-a' }, 'runtime-x')

    await Promise.resolve()

    expect(handled).toBe(true)
    expect(respond).toHaveBeenCalledTimes(1)
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
})

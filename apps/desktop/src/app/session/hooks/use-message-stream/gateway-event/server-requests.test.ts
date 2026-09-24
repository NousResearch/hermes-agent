import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { setActiveSessionId, setSelectedStoredSessionId, setSessions } from '@/store/session'
import { $sessionTiles } from '@/store/session-states'
import { $toursEnabled } from '@/store/tours'
import type { SessionInfo } from '@/types/hermes'

import { handleServerRequest, previewSessionRoute } from './server-requests'
import type { ServerRequestContext } from './server-requests'

const { hudWindow } = vi.hoisted(() => ({ hudWindow: { current: false } }))

vi.mock('@/store/windows', async importOriginal => ({
  ...(await importOriginal()),
  isHudWindow: () => hudWindow.current
}))

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
})

describe('HUD window.read claim', () => {
  const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }

  beforeEach(() => {
    hudWindow.current = true
    // The conversation the HUD is showing, as a stored id — the same source
    // useReportHudSession reports to main — plus the sessions list that maps
    // it back to the live runtime id the gateway names in the request.
    setSelectedStoredSessionId('stored-a')
    setSessions([
      { id: 'rt-tip', title: 'HUD conversation', _lineage_ids: ['rt-old', 'rt-tip'], _lineage_root_id: 'stored-a' } as SessionInfo
    ])
  })

  afterEach(() => {
    hudWindow.current = false
    setSelectedStoredSessionId(null)
    setSessions([])
    delete desktopWindow.hermesDesktop
  })

  it('answers window.read in the HUD even though it never binds the active session', async () => {
    // HUD mode: the transport moved to the HUD but $activeSessionId stays
    // unset on this window (main holds the id on its behalf) — without the
    // claim, the request took route 'ignore' and the tool stalled 30s (#121609).
    const readWindowBelow = vi.fn(async () => ({ window: { app: 'Arc', id: 44083 } }))
    desktopWindow.hermesDesktop = { readWindowBelow } as unknown as Window['hermesDesktop']

    const { handled, respond } = deliver('window.read', { session_id: 'rt-tip' }, null)

    await new Promise(resolve => setTimeout(resolve, 0))

    expect(handled).toBe(true)
    expect(readWindowBelow).toHaveBeenCalledTimes(1)
    expect(respond).toHaveBeenCalledTimes(1)
    expect(JSON.parse(respond.mock.calls[0][0].value)).toMatchObject({ window: { app: 'Arc' } })
  })

  it('still claims when compression rotated the runtime id past the stored one', async () => {
    // The request names a pre-rotation runtime id; lineage matching resolves
    // it to the conversation the HUD is showing.
    desktopWindow.hermesDesktop = { readWindowBelow: async () => null } as unknown as Window['hermesDesktop']

    const { respond } = deliver('window.read', { session_id: 'rt-old' }, null)

    await new Promise(resolve => setTimeout(resolve, 0))

    expect(respond).toHaveBeenCalledTimes(1)
  })

  it('does not claim pane requests the HUD has no pane for', async () => {
    // The HUD hosts the conversation for window.read only: an empty pane
    // answer from the HUD would win the #113348 race and starve the window
    // whose preview/terminal pane is actually open.
    const reads = ['preview.read', 'terminal.read'].map(method => deliver(method, { session_id: 'rt-tip' }, null))

    await new Promise(resolve => setTimeout(resolve, 0))

    for (const { handled, respond } of reads) {
      expect(handled).toBe(true)
      expect(respond).not.toHaveBeenCalled()
    }
  })

  it('keeps a non-HUD window ignoring window.read it does not host', async () => {
    hudWindow.current = false

    const { handled, respond } = deliver('window.read', { session_id: 'rt-tip' }, null)

    await new Promise(resolve => setTimeout(resolve, 0))

    expect(handled).toBe(true)
    expect(respond).not.toHaveBeenCalled()
  })
})

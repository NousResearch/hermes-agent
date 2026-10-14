import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { registerPreviewNav } from '@/app/chat/right-rail/preview-nav'
import { registerPreviewPageReader } from '@/app/chat/right-rail/preview-reader'
import { group } from '@/components/pane-shell/tree/model'
import { $layoutTree, noteActiveTreeGroup } from '@/components/pane-shell/tree/store'
import { createClientSessionState } from '@/lib/chat-runtime'
import { runTour } from '@/lib/tour'
import { $previewTabs, closeRightRail, openPreview, setPreviewTabPinned } from '@/store/preview'
import { hasOpenServerRequest, resetServerRequestsForTests } from '@/store/server-requests'
import { setActiveSessionId, setSelectedStoredSessionId, setSessions } from '@/store/session'
import { $sessionStates, $sessionTiles, dropSessionState, publishSessionState } from '@/store/session-states'
import { $toursEnabled } from '@/store/tours'
import type { SessionInfo } from '@/types/hermes'

import { handleServerRequest, previewSessionRoute, requestNamesActiveSession } from './server-requests'
import type { ServerRequestContext } from './server-requests'

vi.mock('@/lib/tour', () => ({ runTour: vi.fn(async () => ({ ok: true })) }))

const hasLivePreviewSurface = vi.hoisted(() => vi.fn((_owner?: unknown): boolean => false))

const requestPopoutPreviewAct = vi.hoisted(() =>
  vi.fn(async (_payload: unknown, _owner?: unknown, _signal?: AbortSignal): Promise<unknown> => null)
)

const requestPopoutPreviewRead = vi.hoisted(() =>
  vi.fn(async (_payload: unknown, _owner?: unknown): Promise<unknown> => null)
)

vi.mock('@/app/chat/right-rail/preview-popout-bridge', () => ({
  hasLivePreviewSurface: (owner?: unknown) => hasLivePreviewSurface(owner),
  requestPopoutPreviewAct: (payload: unknown, owner?: unknown, signal?: AbortSignal) => requestPopoutPreviewAct(payload, owner, signal),
  requestPopoutPreviewRead: (payload: unknown, owner?: unknown) => requestPopoutPreviewRead(payload, owner)
}))

const deps = {
  activeSessionIdRef: { current: null },
  sessionInterrupted: () => false,
  sessionStateByRuntimeIdRef: { current: new Map() },
  updateSessionState: (_sessionId, update) => update(createClientSessionState('stored-session')),
  upsertToolCall: () => undefined
} as ServerRequestContext['deps']

function deliver(method: string, params: Record<string, unknown>, activeSessionId: null | string, replayed?: boolean) {
  const respond = vi.fn()
  const fail = vi.fn()
  const decline = vi.fn()

  const handled = handleServerRequest(
    { decline, fail, id: 'srq-1', method, params, profile: 'default', replayed, respond },
    deps,
    activeSessionId
  )

  return { decline, fail, handled, respond }
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

  it('declines a scoped action request in a window showing another session instead of answering it', () => {
    // A decline leaves the request open for the owner window; the backend
    // settles only when every attached window declined (#119333, #113348).
    const { decline, handled, respond, fail } = deliver(
      'preview.act',
      { action: 'elements', session_id: 'session-a' },
      'session-b'
    )

    expect(handled).toBe(true)
    expect(decline).toHaveBeenCalledTimes(1)
    expect(respond).not.toHaveBeenCalled()
    expect(fail).not.toHaveBeenCalled()
  })

  it('declines scoped pane reads in a window showing another session', async () => {
    // A decline is not an answer: resolve_response keeps the FIRST response,
    // so a fast empty answer from a non-claiming window could beat the
    // claimant's real answer in the fanout race (review of #121715). The
    // backend counts the decline as that client's vote and keeps the
    // request open for the owner (#119333).
    const reads = ['preview.read', 'terminal.read', 'window.read'].map(method =>
      deliver(method, { session_id: 'session-a' }, 'session-b')
    )

    await Promise.resolve()

    for (const { decline, handled, respond } of reads) {
      expect(handled).toBe(true)
      expect(decline).toHaveBeenCalledTimes(1)
      expect(respond).not.toHaveBeenCalled()
    }
  })

  it('declines a replayed request only after the retry still finds no host', async () => {
    const replay = deliver('preview.read', { session_id: 'session-a' }, null, true)

    expect(replay.decline).not.toHaveBeenCalled()
    await new Promise(resolve => setTimeout(resolve, 0))
    expect(replay.decline).toHaveBeenCalledTimes(1)
    expect(replay.respond).not.toHaveBeenCalled()
  })

  it('declines window.read instead of stalling when no window claims the session (#121609)', async () => {
    // window.read has no per-window pane: an unclaimed request used to wait
    // out the tool's full 30s deadline because silence was the "not mine"
    // signal. It now declines — the backend settles fast once every attached
    // window declined (#119333) — while pane-owned reads keep waiting for
    // their owner, whose real answer must not be beaten by an empty one (#113348).
    const { decline, handled, respond } = deliver('window.read', { session_id: 'session-a' }, 'session-b')

    await Promise.resolve()

    expect(handled).toBe(true)
    expect(decline).toHaveBeenCalledTimes(1)
    expect(respond).not.toHaveBeenCalled()
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

      for (const { decline, handled, respond } of reads) {
        expect(handled).toBe(true)
        expect(respond).toHaveBeenCalledTimes(1)
        expect(decline).not.toHaveBeenCalled()
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
    setSessions([{ id: 'stored-a', title: 'HUD conversation', _lineage_root_id: 'root-a' } as SessionInfo])
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

    expect(
      previewSessionRoute({ activeSessionId: 'runtime-x', method: 'window.read', replayed: false, sessionId: 'root-a' })
    ).toBe('run')
    expect(
      previewSessionRoute({ activeSessionId: null, method: 'window.read', replayed: false, sessionId: 'root-a' })
    ).toBe('run')
    // Plain stored-id ask (no rotation): selected matches directly.
    expect(
      previewSessionRoute({
        activeSessionId: 'runtime-x',
        method: 'window.read',
        replayed: false,
        sessionId: 'stored-a'
      })
    ).toBe('run')
  })

  it('maps an unknown runtime id through the session-state cache to the shown conversation', () => {
    // A resume rebound the runtime without this window re-deriving lineage:
    // the state cache records which stored id the runtime id belongs to.
    setSelectedStoredSessionId('stored-a')
    $sessionStates.set({ 'runtime-rotated': createClientSessionState('stored-a') })

    expect(
      previewSessionRoute({
        activeSessionId: 'runtime-rotated',
        method: 'window.read',
        replayed: false,
        sessionId: 'runtime-rotated'
      })
    ).toBe('run')
  })

  it('claims for a tile whose stored session lineage-matches the asked id', () => {
    $sessionTiles.set([{ runtimeId: 'tile-runtime', storedSessionId: 'stored-a' } as never])

    expect(
      previewSessionRoute({ activeSessionId: 'session-b', method: 'window.read', replayed: false, sessionId: 'root-a' })
    ).toBe('run')
    expect(
      previewSessionRoute({
        activeSessionId: 'session-b',
        method: 'window.read',
        replayed: false,
        sessionId: 'stored-a'
      })
    ).toBe('run')
  })

  it('keeps every other window-owned method on the strict host check even when the identity is tolerated', () => {
    // preview.act and tour refuse on raw isActiveSession, so widening their
    // claim would turn another window's silence into a false refusal that
    // wins the race — and a tour refusal latches session["tour_bridge"],
    // converting later tour actions into 45s waits (review of #121715).
    setSelectedStoredSessionId('stored-a')

    expect(
      previewSessionRoute({
        activeSessionId: 'runtime-x',
        method: 'preview.read',
        replayed: false,
        sessionId: 'root-a'
      })
    ).toBe('ignore')
    expect(
      previewSessionRoute({
        activeSessionId: 'runtime-x',
        method: 'terminal.read',
        replayed: false,
        sessionId: 'root-a'
      })
    ).toBe('ignore')
    expect(
      previewSessionRoute({ activeSessionId: 'runtime-x', method: 'preview.act', replayed: false, sessionId: 'root-a' })
    ).toBe('ignore')
    expect(
      previewSessionRoute({ activeSessionId: 'runtime-x', method: 'tour', replayed: false, sessionId: 'root-a' })
    ).toBe('ignore')
  })

  it('never claims a conversation this window does not show', () => {
    setSelectedStoredSessionId('stored-a')

    expect(
      previewSessionRoute({
        activeSessionId: 'session-b',
        method: 'window.read',
        replayed: false,
        sessionId: 'session-unrelated'
      })
    ).toBe('ignore')
    expect(
      previewSessionRoute({
        activeSessionId: 'session-b',
        method: 'window.read',
        replayed: false,
        sessionId: 'root-other'
      })
    ).toBe('ignore')
  })

  it('claims nothing without a shown conversation — a background session stays unclaimed', () => {
    // No selection, no tiles: the tolerant branch must stay inert so a window
    // midsession cannot answer for a background conversation it never showed.
    expect(
      previewSessionRoute({ activeSessionId: 'session-b', method: 'window.read', replayed: false, sessionId: 'root-a' })
    ).toBe('ignore')
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

describe('preview pop-out forwarding', () => {
  // The requester reaches the pop-out: it answers only for that session's tabs.
  // It also names its profile: only that profile's pins may answer it.
  const sessionAOwner = { profile: 'default', runtimeId: 'session-a', sessionId: 'stored-a' }

  beforeEach(() => {
    hasLivePreviewSurface.mockReturnValue(false)
    requestPopoutPreviewAct.mockClear()
    requestPopoutPreviewAct.mockResolvedValue(null)
    requestPopoutPreviewRead.mockClear()
    requestPopoutPreviewRead.mockResolvedValue(null)
    publishSessionState('session-a', createClientSessionState('stored-a'))
  })

  afterEach(() => {
    dropSessionState('session-a')
  })

  it('forwards an active-session act to the pop-out when this window has no live surface', async () => {
    requestPopoutPreviewAct.mockResolvedValue({ acted: 'elements', success: true })

    const { respond } = deliver('preview.act', { action: 'elements', session_id: 'session-a' }, 'session-a')

    await vi.waitFor(() => expect(respond).toHaveBeenCalledTimes(1), { timeout: 30_000 })
    expect(requestPopoutPreviewAct).toHaveBeenCalledWith(expect.objectContaining({ kind: 'elements' }), sessionAOwner, expect.any(AbortSignal))
    expect(JSON.parse(respond.mock.calls[0][0].value)).toMatchObject({ acted: 'elements', success: true })
  })

  it.each([
    { action: 'drag', selector: 'svg circle', dx: -40.5, dy: 0 },
    { action: 'elements', full: true, max: 7 },
    { action: 'press', ref: 'body', key: 'a', allow_shortcut: true }
  ])('preserves the full validated wire payload: %j', async payload => {
    requestPopoutPreviewAct.mockResolvedValue({ success: true })
    const { respond } = deliver('preview.act', { ...payload, session_id: 'session-a' }, 'session-a')
    await vi.waitFor(() => expect(respond).toHaveBeenCalled())
    const { action, ...fields } = payload
    const mapped = Object.fromEntries(Object.entries(fields).map(([key, value]) => [key === 'allow_shortcut' ? 'allowShortcut' : key, value]))
    expect(requestPopoutPreviewAct).toHaveBeenCalledWith({ kind: action, ...mapped }, sessionAOwner, expect.any(AbortSignal))
  })

  it.each([{ dx: '40' }, { selector: null }, { text: null }, { unknown: true }, { kind: 'click' }])('rejects malformed drag before surface delivery: %j', change => {
    const { respond } = deliver('preview.act', { action: 'drag', selector: '#handle', dx: 40, dy: 20, ...change, session_id: 'session-a' }, 'session-a')
    expect(requestPopoutPreviewAct).not.toHaveBeenCalled()
    expect(JSON.parse(respond.mock.calls[0][0].value)).toMatchObject({ success: false })
  })

  it('never replays an uncertain remote write on a newly available local surface', async () => {
    let resolve!: (value: unknown) => void
    requestPopoutPreviewAct.mockImplementationOnce(() => new Promise(done => { resolve = done }))
    const { respond } = deliver('preview.act', { action: 'drag', selector: '#handle', dx: 40, dy: 20, session_id: 'session-a' }, 'session-a')
    hasLivePreviewSurface.mockReturnValue(true)
    resolve(null)
    await vi.waitFor(() => expect(respond).toHaveBeenCalled())
    expect(JSON.parse(respond.mock.calls[0][0].value)).toMatchObject({ success: false, error: expect.stringContaining('not replayed locally') })
    expect(requestPopoutPreviewAct).toHaveBeenCalledTimes(1)
  })

  it('runs the act locally when this window has a live surface', async () => {
    hasLivePreviewSurface.mockReturnValue(true)

    const { respond } = deliver('preview.act', { action: 'elements', session_id: 'session-a' }, 'session-a')

    await vi.waitFor(() => expect(respond).toHaveBeenCalledTimes(1), { timeout: 30_000 })
    expect(requestPopoutPreviewAct).not.toHaveBeenCalled()
  })

  it('reads from the pop-out when the chat window has no live surface', async () => {
    requestPopoutPreviewRead.mockResolvedValue({ kind: 'url', text: 'page' })

    const { respond } = deliver('preview.read', { count: 100, session_id: 'session-a', start: 0 }, 'session-a')

    await vi.waitFor(() => expect(respond).toHaveBeenCalledTimes(1), { timeout: 30_000 })
    expect(requestPopoutPreviewRead).toHaveBeenCalledWith({ count: 100, start: 0 }, sessionAOwner)
    expect(JSON.parse(respond.mock.calls[0][0].value)).toMatchObject({ kind: 'url', text: 'page' })
  })

  it('falls back to the local read when no pop-out answers', async () => {
    const { respond } = deliver('preview.read', { session_id: 'session-a' }, 'session-a')

    await vi.waitFor(() => expect(respond).toHaveBeenCalledTimes(1), { timeout: 30_000 })
    expect(requestPopoutPreviewRead).toHaveBeenCalledWith({}, sessionAOwner)
    expect(respond.mock.calls[0][0].value).toBe('')
  })
})

describe('preview requests act for the session that asked (#73890)', () => {
  beforeEach(() => {
    hasLivePreviewSurface.mockReturnValue(true)
    closeRightRail()
    setActiveSessionId('rt-a')
    setSelectedStoredSessionId('stored-a')
    $sessionTiles.set([{ dir: 'right', runtimeId: 'rt-b', storedSessionId: 'stored-b' } as never])
  })

  afterEach(() => {
    noteActiveTreeGroup(null)
    $layoutTree.set(null)
    $sessionTiles.set([])
    setActiveSessionId(null)
    setSelectedStoredSessionId(null)
    closeRightRail()
    hasLivePreviewSurface.mockReturnValue(false)
  })

  // Tile B holds focus while the primary session A's agent is the one asking.
  const focusTileB = () => {
    $layoutTree.set(group(['session-tile:stored-b'], { active: 'session-tile:stored-b', id: 'grp-b' }))
    noteActiveTreeGroup('grp-b')
  }

  it("drives the requesting session's page, not the focused tile's", async () => {
    openPreview({ kind: 'url', label: 'a', source: 'https://a.example', url: 'https://a.example' }, 'stored-a')
    openPreview({ kind: 'url', label: 'b', source: 'https://b.example', url: 'https://b.example' }, 'stored-b')
    const [a, b] = $previewTabs.get()
    const backA = vi.fn()
    const backB = vi.fn()

    const unbind = [
      registerPreviewNav(a!.id, { back: backA, forward: () => {}, reload: () => {} }),
      registerPreviewNav(b!.id, { back: backB, forward: () => {}, reload: () => {} })
    ]

    focusTileB()

    try {
      const { respond } = deliver('preview.act', { action: 'back', session_id: 'rt-a' }, 'rt-a')

      await vi.waitFor(() => expect(respond).toHaveBeenCalledTimes(1), { timeout: 30_000 })
      expect(JSON.parse(respond.mock.calls[0][0].value)).toMatchObject({ acted: 'back', success: true })
      expect(backA).toHaveBeenCalledTimes(1)
      expect(backB).not.toHaveBeenCalled()
    } finally {
      unbind.forEach(stop => stop())
    }
  })

  it("runs a preview tour against the requesting session's tabs", async () => {
    focusTileB()
    vi.mocked(runTour).mockClear()

    const { respond } = deliver('tour', { action: 'discover', session_id: 'rt-a', surface: 'preview' }, 'rt-a')

    await vi.waitFor(() => expect(respond).toHaveBeenCalledTimes(1), { timeout: 30_000 })
    expect(runTour).toHaveBeenCalledWith(expect.anything(), 'preview', {
      profile: 'default',
      runtimeId: 'rt-a',
      sessionId: 'stored-a'
    })
  })

  it('drives the tab its runtime opened before the selection named its stored id', async () => {
    setActiveSessionId('rt-new')
    setSelectedStoredSessionId(null)
    publishSessionState('rt-new', createClientSessionState(null))
    openPreview({ kind: 'url', label: 'n', source: 'https://n.example', url: 'https://n.example' }, null, 'rt-new')
    const back = vi.fn()
    const unbind = registerPreviewNav($previewTabs.get()[0]!.id, { back, forward: () => {}, reload: () => {} })
    // The selection names the stored id before rt-new's state binds it.
    setSelectedStoredSessionId('stored-new')

    try {
      const { respond } = deliver('preview.act', { action: 'back', session_id: 'rt-new' }, 'rt-new')

      await vi.waitFor(() => expect(respond).toHaveBeenCalledTimes(1), { timeout: 30_000 })
      expect(JSON.parse(respond.mock.calls[0][0].value)).toMatchObject({ acted: 'back', success: true })
      expect(back).toHaveBeenCalledTimes(1)
    } finally {
      unbind()
      dropSessionState('rt-new')
    }
  })

  it("never answers a tile of another profile with the viewed profile's pinned page", async () => {
    // Profile X's runtime stays open as a tile while the primary's profile
    // (the bucket in view) has a pinned Browser.
    $sessionTiles.set([
      { dir: 'right', ownerProfile: 'prof-p1-x', runtimeId: 'rt-x', storedSessionId: 'stored-x' } as never
    ])
    publishSessionState('rt-x', createClientSessionState('stored-x'))
    openPreview({ kind: 'url', label: 'y', source: 'https://y.example', url: 'https://y.example' }, 'stored-a')
    const pin = $previewTabs.get()[0]!.id
    setPreviewTabPinned(pin, true)

    const unbind = registerPreviewPageReader(pin, async () => ({
      text: 'PROFILE_Y_PRIVATE_TEXT',
      title: 'y',
      url: 'https://y.example'
    }))

    try {
      const fromX = deliver('preview.read', { session_id: 'rt-x' }, 'rt-a')

      await vi.waitFor(() => expect(fromX.respond).toHaveBeenCalledTimes(1), { timeout: 30_000 })
      expect(fromX.respond.mock.calls[0][0].value).toBe('')

      // Control: the pin's own profile still reads it.
      const fromA = deliver('preview.read', { session_id: 'rt-a' }, 'rt-a')

      await vi.waitFor(() => expect(fromA.respond).toHaveBeenCalledTimes(1), { timeout: 30_000 })
      expect(JSON.parse(fromA.respond.mock.calls[0][0].value)).toMatchObject({ text: 'PROFILE_Y_PRIVATE_TEXT' })
    } finally {
      unbind()
      dropSessionState('rt-x')
    }
  })

  it('reads no tab for a requester whose stored id is not bound yet, never the focused one', async () => {
    // A fresh primary runtime whose stored id has not arrived; tile B focused.
    setActiveSessionId('rt-new')
    setSelectedStoredSessionId(null)
    openPreview(
      {
        kind: 'file',
        label: 'b',
        path: '/work/b-secret.txt',
        source: '/work/b-secret.txt',
        url: 'file:///work/b-secret.txt'
      },
      'stored-b'
    )
    focusTileB()

    const { respond } = deliver('preview.read', { session_id: 'rt-new' }, 'rt-new')

    await vi.waitFor(() => expect(respond).toHaveBeenCalledTimes(1), { timeout: 30_000 })
    expect(respond.mock.calls[0][0].value).toBe('')
  })
})

describe('tour request routing', () => {
  afterEach(() => {
    $toursEnabled.set(true)
  })

  it('declines a scoped request in another session even when tours are disabled', () => {
    $toursEnabled.set(false)
    const { decline, handled, respond } = deliver('tour', { action: 'discover', session_id: 'session-a' }, 'session-b')

    expect(handled).toBe(true)
    expect(decline).toHaveBeenCalledTimes(1)
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
      await vi.waitFor(() => expect(respond).toHaveBeenCalledTimes(1), { timeout: 30_000 })
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

// #75587: a blocking-input request still in flight when the session's runtime is
// interrupted (Stop) or deleted must not park its card — parking one would
// resurrect an overlay (and native notification) for a turn that is gone. It is
// answered with an error (the backend's "unanswered"), not dropped, so the
// blocked tool returns instead of waiting out its deadline.
describe('blocking-input guard for interrupted sessions', () => {
  const depsWith = (interrupted: boolean) =>
    ({ ...deps, sessionInterrupted: () => interrupted }) as ServerRequestContext['deps']

  const approvalRequest = (id: string) => ({
    fail: vi.fn(),
    id,
    method: 'approval',
    params: { command: 'rm -rf /', description: 'dangerous', request_id: 'r1', session_id: 'session-a' },
    profile: 'default',
    respond: vi.fn()
  })

  afterEach(() => {
    resetServerRequestsForTests()
  })

  it('fails an approval request for an interrupted session instead of parking it', () => {
    const request = approvalRequest('srq-dead')

    expect(handleServerRequest(request, depsWith(true), 'session-a')).toBe(true)

    expect(hasOpenServerRequest('srq-dead')).toBe(false)
    expect(request.fail).toHaveBeenCalledWith(expect.any(Number), 'session interrupted')
    expect(request.respond).not.toHaveBeenCalled()
  })

  it('still parks an approval request for a live session', () => {
    const request = approvalRequest('srq-live')

    handleServerRequest(request, depsWith(false), 'session-a')

    expect(hasOpenServerRequest('srq-live')).toBe(true)
    expect(request.fail).not.toHaveBeenCalled()
  })
})

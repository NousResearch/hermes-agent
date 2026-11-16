import type { ServerRequest } from '@hermes/shared/json-rpc-channel'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { createBackgroundSessionWatch } from '../app/backgroundSessionNotices.js'
import { createServerRequestHandler } from '../app/createServerRequestHandler.js'
import { createSlashHandler } from '../app/createSlashHandler.js'
import { getOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'

// /new mid-turn opens a fresh session and leaves the running one working in the background,
// reachable from the session switcher. /clear still refuses to replace a running turn.
function slashCtx() {
  const session = {
    closeSession: vi.fn(() => Promise.resolve(null)),
    die: vi.fn(),
    dieWithCode: vi.fn(),
    guardBusySessionSwitch: vi.fn(() => true),
    newLiveSession: vi.fn(),
    newSession: vi.fn(),
    resetVisibleHistory: vi.fn(),
    resumeById: vi.fn(),
    setSessionStartedAt: vi.fn()
  }

  const request = vi.fn(async () => ({}))

  const ctx = {
    composer: { enqueue: vi.fn() },
    gateway: { gw: { isCanonical: false, request }, rpc: request },
    local: {
      catalog: null,
      getHistoryItems: () => [],
      getLastUserMsg: () => '',
      maybeWarn: vi.fn(),
      setCatalog: vi.fn()
    },
    session,
    slashFlightRef: { current: 0 },
    transcript: {
      page: vi.fn(),
      panel: vi.fn(),
      send: vi.fn(),
      setHistoryItems: vi.fn(),
      sys: vi.fn(),
      trimLastExchange: (x: unknown[]) => x
    }
  }

  return { ctx, session }
}

const serverRequest = (sessionId: string, replayed = false) => {
  const respond = vi.fn()

  return {
    request: {
      fail: vi.fn(),
      id: `srq-${sessionId}`,
      method: 'approval',
      params: { command: 'rm -rf build', session_id: sessionId },
      replayed,
      respond
    } as unknown as ServerRequest,
    respond
  }
}

describe('/new keeps a running session alive in the background', () => {
  beforeEach(() => {
    resetOverlayState()
    resetUiState()
    patchUiState({ busy: true, sid: 'sid-running' })
  })

  it('starts a new live session mid-turn without the busy guard or a destructive confirm', () => {
    const { ctx, session } = slashCtx()

    expect(createSlashHandler(ctx as never)('/new side quest')).toBe(true)

    expect(session.newLiveSession).toHaveBeenCalledWith(expect.any(String), 'side quest')
    expect(session.newSession).not.toHaveBeenCalled()
    expect(session.guardBusySessionSwitch).not.toHaveBeenCalled()
    expect(getOverlayState().confirm).toBeNull()
  })

  it('still guards /clear mid-turn, which replaces the session', () => {
    const { ctx, session } = slashCtx()

    createSlashHandler(ctx as never)('/clear')

    expect(session.guardBusySessionSwitch).toHaveBeenCalled()
    expect(session.newLiveSession).not.toHaveBeenCalled()
    expect(session.newSession).not.toHaveBeenCalled()
  })
})

describe("a background session's prompts wait for its own view", () => {
  beforeEach(() => {
    resetOverlayState()
    resetUiState()
  })

  it('does not open another session’s approval over the focused one, and opens it once activated', () => {
    let focused = 'sid-new'

    const handle = createServerRequestHandler({
      focusedSessionId: () => focused,
      ringPromptBell: vi.fn(),
      setStatus: vi.fn()
    })

    const live = serverRequest('sid-background')

    expect(handle(live.request)).toBe(true)
    expect(getOverlayState().approval).toBeNull()
    expect(live.respond).not.toHaveBeenCalled() // still open on the backend, never auto-answered

    // Activating the background session re-delivers its open request (`open_requests`, replayed).
    focused = 'sid-background'
    handle(serverRequest('sid-background', true).request)
    expect(getOverlayState().approval?.command).toBe('rm -rf build')
  })

  it('announces a shown background session finishing its turn, never the focused one', () => {
    const watch = createBackgroundSessionWatch()

    watch.focus('sid-old')
    watch.focus('sid-new')
    expect(
      watch.update(
        [
          { id: 'sid-old', status: 'working' },
          { id: 'sid-new', status: 'idle' }
        ],
        'sid-new'
      )
    ).toEqual([])

    const notices = watch.update(
      [
        { id: 'sid-old', status: 'idle', title: 'refactor' },
        { id: 'sid-new', status: 'working' }
      ],
      'sid-new'
    )

    expect(notices).toHaveLength(1)
    expect(notices[0]).toContain('refactor')
  })
})

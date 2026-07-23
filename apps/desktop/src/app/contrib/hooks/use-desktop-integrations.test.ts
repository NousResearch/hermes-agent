import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useDesktopIntegrations } from './use-desktop-integrations'

const requestComposerFocus = vi.fn()
const requestComposerInsert = vi.fn()

vi.mock('@/app/chat/close-tab', () => ({ closeActiveTab: vi.fn() }))
vi.mock('@/store/native-notifications', () => ({ respondToApprovalAction: vi.fn() }))
vi.mock('@/store/session', () => ({
  getRememberedRoute: vi.fn(() => null),
  getRememberedSessionId: vi.fn(() => null),
  setRememberedRoute: vi.fn(),
  setRememberedSessionId: vi.fn()
}))
vi.mock('@/store/session-sync', () => ({ onSessionsChanged: vi.fn(() => () => {}) }))
vi.mock('@/store/updates', () => ({
  openUpdatesWindow: vi.fn(),
  startUpdatePoller: vi.fn(),
  stopUpdatePoller: vi.fn()
}))
vi.mock('@/store/windows', () => ({ isSecondaryWindow: vi.fn(() => true) }))
vi.mock('@/app/chat/composer/focus', () => ({
  requestComposerFocus: (...args: unknown[]) => requestComposerFocus(...args),
  requestComposerInsert: (...args: unknown[]) => requestComposerInsert(...args)
}))

interface DeepLinkPayload {
  kind: string
  name: string
  params: Record<string, string>
}

let deepLinkListeners: Array<(payload: DeepLinkPayload) => void> = []
let listenersAtReadySignal = -1

// Snapshot how many deep-link listeners were live at the moment the renderer
// signalled readiness — electron/main.ts flushes a cold-start-queued link as
// soon as this resolves, so every kind's listener must already be mounted.
const signalDeepLinkReady = vi.fn(async () => {
  listenersAtReadySignal = deepLinkListeners.length

  return { ok: true }
})

beforeEach(() => {
  deepLinkListeners = []
  listenersAtReadySignal = -1
  ;(window as { hermesDesktop?: unknown }).hermesDesktop = {
    onDeepLink: (callback: (payload: DeepLinkPayload) => void) => {
      deepLinkListeners.push(callback)

      return () => {
        deepLinkListeners = deepLinkListeners.filter(listener => listener !== callback)
      }
    },
    signalDeepLinkReady
  }
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  delete (window as { hermesDesktop?: unknown }).hermesDesktop
})

function mountIntegrations() {
  const navigate = vi.fn()

  renderHook(() =>
    useDesktopIntegrations({
      chatOpen: true,
      hasPreview: false,
      locationPathname: '/skills',
      navigate,
      refreshSessions: vi.fn(),
      resumeExhaustedSessionId: null,
      routedSessionId: null,
      runtimeIdByStoredSessionId: { current: new Map() }
    })
  )

  return navigate
}

// Fan a payload out exactly like the preload bridge does: every subscribed
// listener sees every 'hermes:deep-link' IPC message.
function emitDeepLink(payload: DeepLinkPayload) {
  act(() => {
    for (const listener of [...deepLinkListeners]) {
      listener(payload)
    }
  })
}

describe('useDesktopIntegrations deep links', () => {
  it('opens the session for a well-formed hermes://session/<id> link', () => {
    const navigate = mountIntegrations()

    emitDeepLink({ kind: 'session', name: '20260718_101530_ab12cd', params: {} })

    expect(navigate).toHaveBeenCalledTimes(1)
    expect(navigate).toHaveBeenCalledWith('/20260718_101530_ab12cd')
  })

  it('ignores malformed session ids silently', () => {
    const navigate = mountIntegrations()

    emitDeepLink({ kind: 'session', name: '', params: {} })
    emitDeepLink({ kind: 'session', name: 'not-a-session-id', params: {} })
    emitDeepLink({ kind: 'session', name: '20260718_101530_ab12cd/../oops', params: {} })
    emitDeepLink({ kind: 'session', name: '20260718_101530_AB12CD', params: {} })

    expect(navigate).not.toHaveBeenCalled()
    expect(requestComposerInsert).not.toHaveBeenCalled()
  })

  it('leaves blueprint links to the composer handler, untouched by the session one', () => {
    const navigate = mountIntegrations()

    emitDeepLink({ kind: 'blueprint', name: 'morning-brief', params: { time: '08:00' } })

    expect(requestComposerInsert).toHaveBeenCalledWith('/blueprint morning-brief time=08:00', {
      mode: 'block',
      target: 'main'
    })
    expect(requestComposerFocus).toHaveBeenCalledWith('main')
    expect(navigate).not.toHaveBeenCalled()
  })

  it('ignores unknown kinds entirely', () => {
    const navigate = mountIntegrations()

    emitDeepLink({ kind: 'skill', name: '20260718_101530_ab12cd', params: {} })

    expect(navigate).not.toHaveBeenCalled()
    expect(requestComposerInsert).not.toHaveBeenCalled()
  })

  it('mounts both deep-link listeners before signalling ready, so a cold-start-queued link flushes into them', () => {
    const navigate = mountIntegrations()

    expect(signalDeepLinkReady).toHaveBeenCalledTimes(1)
    expect(listenersAtReadySignal).toBe(2)

    // The flush replays the queued link through the same channel.
    emitDeepLink({ kind: 'session', name: '20260101_000000_0a0b0c', params: {} })

    expect(navigate).toHaveBeenCalledWith('/20260101_000000_0a0b0c')
  })
})

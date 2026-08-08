import { renderHook } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useDesktopIntegrations } from './use-desktop-integrations'

// The hook wires up a dozen desktop integrations; this file is only about the
// `hermes://` deep-link one. Everything it reaches for on mount is stubbed so
// the effect under test runs in isolation rather than dragging in the update
// poller, the session store and the notification bridge.
vi.mock('@/store/updates', () => ({
  openUpdatesWindow: vi.fn(),
  startUpdatePoller: vi.fn(),
  stopUpdatePoller: vi.fn()
}))
vi.mock('@/store/session-sync', () => ({ onSessionsChanged: vi.fn(() => () => undefined) }))
vi.mock('@/store/native-notifications', () => ({ respondToApprovalAction: vi.fn() }))
vi.mock('@/store/projects', () => ({ openFolderAsProject: vi.fn() }))
vi.mock('@/store/windows', () => ({ isSecondaryWindow: () => false }))
vi.mock('@/app/chat/close-tab', () => ({ closeActiveTab: vi.fn() }))
vi.mock('@/app/open-session', () => ({ openSession: vi.fn() }))

const { openSession } = await import('@/app/open-session')
vi.mock('@/lib/session-ids', () => ({ storedSessionIdForNotification: vi.fn() }))
vi.mock('../../chat/composer/focus', () => ({
  requestComposerFocus: vi.fn(),
  requestComposerInsert: vi.fn()
}))

type DeepLinkPayload = { kind: string; name: string; params?: Record<string, string> }

/** Captures the deep-link subscriber the hook registers, so a test can fire it. */
function mountWithDeepLink(navigate: (to: string, options?: { replace?: boolean }) => void) {
  let deliver: ((payload: DeepLinkPayload) => void) | undefined

  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      onDeepLink: (cb: (payload: DeepLinkPayload) => void) => {
        deliver = cb

        return () => undefined
      },
      signalDeepLinkReady: vi.fn()
    },
    writable: true
  })

  renderHook(() =>
    useDesktopIntegrations({
      activeProfile: 'default',
      chatOpen: false,
      hasPreview: false,
      locationPathname: '/',
      navigate,
      profileReady: true,
      refreshSessions: vi.fn(),
      resumeExhaustedSessionId: null,
      routedSessionId: null,
      runtimeIdByStoredSessionId: { current: new Map<string, string>() },
      sessions: []
    })
  )

  if (deliver === undefined) {
    throw new Error('the hook never subscribed to deep links')
  }

  return deliver
}

describe('hermes:// session deep links', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  // `openSession`, not a bare `navigate` — the distinction is the whole bug this
  // covers. Navigating alone changes the address without selecting the session,
  // which renders an empty pane until something else forces the selection.
  it('opens the named session through openSession', () => {
    const navigate = vi.fn()
    mountWithDeepLink(navigate)({ kind: 'session', name: '20260804_184317_5b179b' })

    expect(openSession).toHaveBeenCalledWith('20260804_184317_5b179b', navigate)
  })

  it('does not route without selecting — a bare navigate would leave an empty pane', () => {
    const navigate = vi.fn()
    mountWithDeepLink(navigate)({ kind: 'session', name: '20260804_184317_5b179b' })

    expect(navigate).not.toHaveBeenCalled()
  })

  it('ignores a session link with no id', () => {
    const navigate = vi.fn()
    mountWithDeepLink(navigate)({ kind: 'session', name: '' })

    expect(openSession).not.toHaveBeenCalled()
  })

  it('leaves other kinds alone — a blueprint link must not open a session', () => {
    const navigate = vi.fn()
    mountWithDeepLink(navigate)({ kind: 'blueprint', name: 'morning-brief' })

    expect(openSession).not.toHaveBeenCalled()
  })
})

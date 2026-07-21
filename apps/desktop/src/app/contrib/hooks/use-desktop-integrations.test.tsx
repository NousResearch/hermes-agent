import { cleanup, render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getRememberedRoute, getRememberedSessionId, setRememberedRoute, setRememberedSessionId } from '@/store/session'

import { NEW_CHAT_ROUTE, sessionRoute } from '../../routes'

import { useDesktopIntegrations } from './use-desktop-integrations'

vi.mock('@/app/chat/close-tab', () => ({ closeActiveTab: vi.fn() }))
vi.mock('@/store/native-notifications', () => ({ respondToApprovalAction: vi.fn() }))
vi.mock('@/store/session-sync', () => ({ onSessionsChanged: vi.fn(() => vi.fn()) }))
vi.mock('@/store/updates', () => ({
  openUpdatesWindow: vi.fn(),
  startUpdatePoller: vi.fn(),
  stopUpdatePoller: vi.fn()
}))
vi.mock('@/store/windows', () => ({ isSecondaryWindow: vi.fn(() => false) }))
vi.mock('../../chat/composer/focus', () => ({
  requestComposerFocus: vi.fn(),
  requestComposerInsert: vi.fn()
}))

interface HarnessProps {
  locationPathname: string
  navigate: (to: string, options?: { replace?: boolean }) => void
  routedSessionId?: null | string
}

const makeNavigate = () => vi.fn<(to: string, options?: { replace?: boolean }) => void>()

function Harness({ locationPathname, navigate, routedSessionId = null }: HarnessProps) {
  useDesktopIntegrations({
    chatOpen: true,
    hasPreview: false,
    locationPathname,
    navigate,
    refreshSessions: vi.fn(),
    resumeExhaustedSessionId: null,
    routedSessionId,
    runtimeIdByStoredSessionId: { current: new Map() }
  })

  return null
}

describe('useDesktopIntegrations startup restore', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.clearAllMocks()
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {}
    })
  })

  afterEach(() => {
    cleanup()
    window.localStorage.clear()
  })

  it('restores the captured page route before the default route can overwrite it', async () => {
    setRememberedRoute('/skills')
    setRememberedSessionId('session-1')
    const navigate = makeNavigate()

    render(<Harness locationPathname={NEW_CHAT_ROUTE} navigate={navigate} />)

    await waitFor(() => {
      expect(navigate).toHaveBeenCalledWith('/skills', { replace: true })
    })
  })

  it('falls back to the captured session when no page route was remembered', async () => {
    setRememberedRoute(null)
    setRememberedSessionId('session-1')
    const navigate = makeNavigate()

    render(<Harness locationPathname={NEW_CHAT_ROUTE} navigate={navigate} />)

    await waitFor(() => {
      expect(navigate).toHaveBeenCalledWith(sessionRoute('session-1'), { replace: true })
    })
  })

  it('honors a remembered fresh chat instead of falling back to a stale session', async () => {
    setRememberedRoute(NEW_CHAT_ROUTE)
    setRememberedSessionId('session-1')
    const navigate = makeNavigate()

    render(<Harness locationPathname={NEW_CHAT_ROUTE} navigate={navigate} />)

    await waitFor(() => {
      expect(getRememberedSessionId()).toBeNull()
    })
    expect(navigate).not.toHaveBeenCalled()
  })

  it('clears the remembered session when the user explicitly opens a fresh chat', async () => {
    const navigate = makeNavigate()

    const view = render(
      <Harness locationPathname={sessionRoute('session-1')} navigate={navigate} routedSessionId="session-1" />
    )

    await waitFor(() => {
      expect(getRememberedSessionId()).toBe('session-1')
    })

    view.rerender(<Harness locationPathname={NEW_CHAT_ROUTE} navigate={navigate} />)

    await waitFor(() => {
      expect(getRememberedRoute()).toBe(NEW_CHAT_ROUTE)
      expect(getRememberedSessionId()).toBeNull()
    })
    expect(navigate).not.toHaveBeenCalled()
  })

  it('leaves an empty fresh startup alone', async () => {
    const navigate = makeNavigate()

    render(<Harness locationPathname={NEW_CHAT_ROUTE} navigate={navigate} />)

    await waitFor(() => {
      expect(getRememberedRoute()).toBe(NEW_CHAT_ROUTE)
    })
    expect(navigate).not.toHaveBeenCalled()
  })

  it('does not replace an explicit non-default startup route', async () => {
    setRememberedRoute('/agents')
    setRememberedSessionId('session-1')
    const navigate = makeNavigate()

    render(<Harness locationPathname="/skills" navigate={navigate} />)

    await waitFor(() => {
      expect(getRememberedRoute()).toBe('/skills')
    })
    expect(navigate).not.toHaveBeenCalled()
  })
})

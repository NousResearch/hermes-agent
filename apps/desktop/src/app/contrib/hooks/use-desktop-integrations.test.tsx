import { renderHook, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type * as SessionStore from '@/store/session'

import { NEW_CHAT_ROUTE } from '../../routes'

import { useDesktopIntegrations } from './use-desktop-integrations'

const sessionMocks = vi.hoisted(() => ({
  getRememberedRoute: vi.fn(() => '/session/stored'),
  getRememberedSessionId: vi.fn(() => 'stored-session'),
  setRememberedRoute: vi.fn(),
  setRememberedSessionId: vi.fn()
}))

const updatesMocks = vi.hoisted(() => ({
  openUpdatesWindow: vi.fn(),
  startUpdatePoller: vi.fn(),
  stopUpdatePoller: vi.fn()
}))

vi.mock('@/store/session', async () => {
  const actual = (await vi.importActual('@/store/session')) as typeof SessionStore

  return {
    ...actual,
    ...sessionMocks
  }
})
vi.mock('@/store/updates', () => updatesMocks)
vi.mock('@/store/windows', () => ({
  isNewSessionWindow: vi.fn(() => true),
  isSecondaryWindow: vi.fn(() => false)
}))

function setHermesDesktopBridge() {
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      setPreviewShortcutActive: vi.fn()
    }
  })
}

afterEach(() => {
  vi.clearAllMocks()
})

describe('useDesktopIntegrations', () => {
  it('does not restore remembered session state in a new-session window', async () => {
    setHermesDesktopBridge()
    const navigate = vi.fn()

    renderHook(() =>
      useDesktopIntegrations({
        chatOpen: true,
        hasPreview: false,
        locationPathname: NEW_CHAT_ROUTE,
        navigate,
        refreshSessions: vi.fn(),
        resumeExhaustedSessionId: null,
        routedSessionId: null,
        runtimeIdByStoredSessionId: { current: new Map() }
      })
    )

    await waitFor(() => {
      expect(updatesMocks.startUpdatePoller).toHaveBeenCalledTimes(1)
    })

    expect(sessionMocks.getRememberedRoute).not.toHaveBeenCalled()
    expect(sessionMocks.getRememberedSessionId).not.toHaveBeenCalled()
    expect(sessionMocks.setRememberedRoute).not.toHaveBeenCalled()
    expect(sessionMocks.setRememberedSessionId).not.toHaveBeenCalled()
    expect(navigate).not.toHaveBeenCalled()
  })
})

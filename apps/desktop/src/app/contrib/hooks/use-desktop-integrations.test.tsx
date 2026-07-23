import { renderHook, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  getRememberedRoute: vi.fn<() => null | string>(),
  getRememberedSessionId: vi.fn<() => null | string>(),
  getSession: vi.fn(),
  navigate: vi.fn(),
  setRememberedRoute: vi.fn(),
  setRememberedSessionId: vi.fn()
}))

vi.mock('@/hermes', () => ({ getSession: mocks.getSession }))
vi.mock('@/store/session', () => ({
  getRememberedRoute: mocks.getRememberedRoute,
  getRememberedSessionId: mocks.getRememberedSessionId,
  setRememberedRoute: mocks.setRememberedRoute,
  setRememberedSessionId: mocks.setRememberedSessionId
}))
vi.mock('@/store/updates', () => ({ openUpdatesWindow: vi.fn(), startUpdatePoller: vi.fn(), stopUpdatePoller: vi.fn() }))
vi.mock('@/store/session-sync', () => ({ onSessionsChanged: vi.fn() }))
vi.mock('@/store/native-notifications', () => ({ respondToApprovalAction: vi.fn() }))
vi.mock('@/store/windows', () => ({ isSecondaryWindow: () => true }))
vi.mock('@/app/chat/close-tab', () => ({ closeActiveTab: vi.fn() }))
vi.mock('@/lib/session-ids', () => ({ storedSessionIdForNotification: (id: string) => id }))
vi.mock('../../chat/composer/focus', () => ({ requestComposerFocus: vi.fn(), requestComposerInsert: vi.fn() }))

import { useDesktopIntegrations } from './use-desktop-integrations'

const renderIntegrations = (routedSessionId: null | string) =>
  renderHook(() =>
    useDesktopIntegrations({
      chatOpen: true,
      hasPreview: false,
      locationPathname: '/',
      navigate: mocks.navigate,
      refreshSessions: vi.fn(),
      resumeExhaustedSessionId: null,
      routedSessionId,
      runtimeIdByStoredSessionId: { current: new Map() }
    })
  )

describe('useDesktopIntegrations remembered sessions', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.getRememberedRoute.mockReturnValue(null)
    mocks.getRememberedSessionId.mockReturnValue(null)
  })

  it('does not need the sidebar list to replace a routed delegate child with its parent', async () => {
    mocks.getSession.mockResolvedValue({ id: 'child', source: 'subagent', parent_session_id: 'parent' })

    renderIntegrations('child')

    await waitFor(() => expect(mocks.setRememberedSessionId).toHaveBeenCalledWith('parent'))
    expect(mocks.setRememberedRoute).toHaveBeenCalledWith('/parent')
  })

  it('repairs a child route already persisted before cold start', async () => {
    mocks.getRememberedRoute.mockReturnValue('/child')
    mocks.getSession.mockResolvedValue({ id: 'child', source: 'subagent', parent_session_id: 'parent' })

    renderIntegrations(null)

    await waitFor(() => expect(mocks.navigate).toHaveBeenCalledWith('/parent', { replace: true }))
    expect(mocks.setRememberedSessionId).toHaveBeenCalledWith('parent')
    expect(mocks.setRememberedRoute).toHaveBeenCalledWith('/parent')
  })
})

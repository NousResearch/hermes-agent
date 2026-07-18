import { renderHook } from '@testing-library/react'
import { beforeEach, expect, test, vi } from 'vitest'

import { NEW_CHAT_ROUTE } from '../../routes'

import { useDesktopIntegrations } from './use-desktop-integrations'

const sessionStore = vi.hoisted(() => ({
  $sessions: { get: vi.fn(() => []) },
  getRememberedRoute: vi.fn<() => null | string>(),
  getRememberedSessionId: vi.fn<(_profile?: null | string) => null | string>(),
  rememberedSessionProfile: vi.fn((_sessions: unknown[], _sessionId: null | string, profile: null | string) => profile),
  setRememberedRoute: vi.fn(),
  setRememberedSessionId: vi.fn()
}))

const profileStore = vi.hoisted(() => ({
  $activeGatewayProfile: { get: vi.fn(() => 'default') }
}))

vi.mock('@/app/chat/close-tab', () => ({ closeActiveTab: vi.fn() }))
vi.mock('@/store/native-notifications', () => ({ respondToApprovalAction: vi.fn() }))
vi.mock('@/store/profile', () => profileStore)
vi.mock('@/store/session', () => sessionStore)
vi.mock('@/store/session-sync', () => ({ onSessionsChanged: vi.fn(() => vi.fn()) }))
vi.mock('@/store/updates', () => ({
  openUpdatesWindow: vi.fn(),
  startUpdatePoller: vi.fn(),
  stopUpdatePoller: vi.fn()
}))
vi.mock('@/store/windows', () => ({ isSecondaryWindow: vi.fn(() => false) }))

beforeEach(() => {
  vi.clearAllMocks()
  sessionStore.getRememberedRoute.mockReturnValue(NEW_CHAT_ROUTE)
  sessionStore.getRememberedSessionId.mockReturnValue('old-session')
})

test('an explicitly remembered new chat clears stale session identity instead of restoring it', () => {
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

  expect(sessionStore.setRememberedSessionId).toHaveBeenCalledWith(null, 'default')
  expect(navigate).not.toHaveBeenCalled()
})

test('restoring a remembered session does not clear its identity during the initial new-route render', () => {
  const navigate = vi.fn()

  sessionStore.getRememberedRoute.mockReturnValue('/old-session')

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

  expect(navigate).toHaveBeenCalledWith('/old-session', { replace: true })
  expect(sessionStore.setRememberedSessionId).not.toHaveBeenCalledWith(null, 'default')
})

test('a fresh draft clears only the active profile remembered session', () => {
  const navigate = vi.fn()

  profileStore.$activeGatewayProfile.get.mockReturnValue('work')

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

  expect(sessionStore.setRememberedSessionId).toHaveBeenCalledWith(null, 'work')
})

/**
 * Remembered-route restore vs late disk plugins. A remembered plugin page is
 * session-shaped until its route registers; if the session list arrives first
 * (an already-running backend), the restore used to classify it as a stale
 * session id, navigate to the last chat, and erase the remembered route for
 * every later boot. The latch now also waits for the disk door's first scan.
 */
import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'
import { $diskPluginsScanPending } from '@/contrib/runtime-loader'
import { _resetLegacyDiscardForTests, getRememberedRoute, setRememberedRoute, setRememberedSessionId } from '@/store/session'

import { makeSessionInfo } from '../../../test/session-info'

import { useDesktopIntegrations } from './use-desktop-integrations'

vi.mock('@/store/windows', async importOriginal => ({ ...(await importOriginal<object>()), isHudWindow: () => false }))

const desktopWindow = window as unknown as { hermesDesktop: unknown }
const originalBridge = desktopWindow.hermesDesktop

beforeEach(() => {
  window.localStorage.clear()
  _resetLegacyDiscardForTests()
  desktopWindow.hermesDesktop = {
    setPreviewShortcutActive: vi.fn(),
    onOpenUpdatesRequested: vi.fn(),
    onFocusSession: vi.fn(),
    onNotificationAction: vi.fn(),
    onNotificationActivate: vi.fn(),
    onDeepLink: vi.fn(),
    signalDeepLinkReady: vi.fn(),
    onClosePreviewRequested: vi.fn(),
    onOpenFolderRequested: vi.fn()
  }
})

afterEach(() => {
  $diskPluginsScanPending.set(false)
  desktopWindow.hermesDesktop = originalBridge
})

function mountRestore() {
  const navigate = vi.fn()
  const sessions = [makeSessionInfo({ id: 'sess-1' })]
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
      resumeLastSession: true,
      routedSessionId: null,
      runtimeIdByStoredSessionId: { current: new Map() },
      sessions
    })
  )

  return navigate
}

it('restores a remembered plugin page whose route registers after the session list arrived', () => {
  setRememberedRoute('/html-gallery', 'default')
  setRememberedSessionId('sess-1', 'default')
  $diskPluginsScanPending.set(true)

  const navigate = mountRestore()

  // Latch held: neither the last chat nor an erased route.
  expect(navigate).not.toHaveBeenCalled()
  expect(getRememberedRoute('default')).toBe('/html-gallery')

  const dispose = registry.register({ area: 'routes', id: 'gallery:page', data: { path: '/html-gallery' }, render: () => null })
  act(() => $diskPluginsScanPending.set(false))

  expect(navigate).toHaveBeenCalledWith('/html-gallery', { replace: true })
  expect(getRememberedRoute('default')).toBe('/html-gallery')
  dispose()
})

it('still drops a remembered session route nobody owns once the disk scan has settled', () => {
  setRememberedRoute('/gone-session', 'default')
  setRememberedSessionId('sess-1', 'default')
  $diskPluginsScanPending.set(true)

  const navigate = mountRestore()
  expect(navigate).not.toHaveBeenCalled()

  act(() => $diskPluginsScanPending.set(false))

  expect(navigate).toHaveBeenCalledWith('/sess-1', { replace: true })
  expect(getRememberedRoute('default')).toBeNull()
})

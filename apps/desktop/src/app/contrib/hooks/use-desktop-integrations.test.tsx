import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $activeGatewayProfile, $profileInitialized } from '@/store/profile'
import { stopUpdatePoller } from '@/store/updates'

import { NEW_CHAT_ROUTE, sessionRoute } from '../../routes'

import { useDesktopIntegrations } from './use-desktop-integrations'

// Regression coverage for the cold-start restore race (issue #74387): the
// restore effect must not read $activeGatewayProfile until adoptPrimaryProfile()
// has resolved the real profile from the Electron backend. On mount the atom is
// still the placeholder 'default', and reading profile-scoped localStorage keys
// under 'default' restores the stale global key instead of the named profile's
// scoped route/session.
describe('useDesktopIntegrations cold-start restore', () => {
  function baseProps(
    navigate: (to: string, options?: { replace?: boolean }) => void
  ): Parameters<typeof useDesktopIntegrations>[0] {
    return {
      chatOpen: false,
      hasPreview: false,
      locationPathname: NEW_CHAT_ROUTE,
      navigate,
      refreshSessions: vi.fn(),
      resumeExhaustedSessionId: null,
      routedSessionId: null,
      runtimeIdByStoredSessionId: { current: new Map<string, string>() }
    }
  }

  beforeEach(() => {
    localStorage.clear()
    $activeGatewayProfile.set('default')
    $profileInitialized.set(false)
  })

  afterEach(() => {
    cleanup()
    stopUpdatePoller()
    localStorage.clear()
  })

  it('waits for profile initialization, then restores the named profile\'s scoped route', () => {
    // The named profile's prior run remembered a session route under its scoped
    // key. The global keys hold pre-scoping leftovers that an un-gated restore
    // would wrongly read at 'default'. (On mount the remember-route WRITE
    // effect — declared before this one — runs first and overwrites the global
    // route key with '/', so the unfixed code's leak surfaces via the stale
    // global session id below; the scoped keys are untouched either way.)
    localStorage.setItem('hermes.desktop.lastRoute', '/skills')
    localStorage.setItem('hermes.desktop.lastRoute.atlas', '/session/atlas-session')
    localStorage.setItem('hermes.desktop.lastSessionId', 'legacy-session')

    const navigate = vi.fn()
    renderHook(() => useDesktopIntegrations(baseProps(navigate)))

    // Mounted with the placeholder 'default' profile: the restore must wait.
    expect(navigate).not.toHaveBeenCalled()

    // adoptPrimaryProfile() resolves the real profile after an IPC round-trip.
    act(() => {
      $activeGatewayProfile.set('atlas')
      $profileInitialized.set(true)
    })

    expect(navigate).toHaveBeenCalledTimes(1)
    expect(navigate).toHaveBeenCalledWith('/session/atlas-session', { replace: true })
  })

  it('falls back to the named profile\'s scoped session id, not a stale global one', () => {
    // The route fallback path: no scoped route remembered, but the named
    // profile did remember a session. The global id is a stale pre-scoping
    // leftover that the un-gated restore would navigate to instead.
    localStorage.setItem('hermes.desktop.lastSessionId', 'legacy-session')
    localStorage.setItem('hermes.desktop.lastSessionId.atlas', 'atlas-session')

    // Profile already resolved before the integrations mount (HMR survivor /
    // adoptBoot path) — restore must run immediately on the correct profile.
    $activeGatewayProfile.set('atlas')
    $profileInitialized.set(true)

    const navigate = vi.fn()
    renderHook(() => useDesktopIntegrations(baseProps(navigate)))

    expect(navigate).toHaveBeenCalledTimes(1)
    expect(navigate).toHaveBeenCalledWith(sessionRoute('atlas-session'), { replace: true })
  })

  it('never restores when the window started away from the new-chat route', () => {
    // A hidden-then-shown window keeps its own route: profile resolution later
    // must not trigger a restore into the remembered route.
    localStorage.setItem('hermes.desktop.lastRoute.atlas', '/session/atlas-session')
    $activeGatewayProfile.set('atlas')

    const navigate = vi.fn()
    renderHook(() => useDesktopIntegrations({ ...baseProps(navigate), locationPathname: '/skills' }))

    act(() => {
      $profileInitialized.set(true)
    })

    expect(navigate).not.toHaveBeenCalled()
  })
})

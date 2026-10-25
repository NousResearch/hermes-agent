import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { installBrowserDesktopBridge } from '@/lib/browser-desktop-bridge'
import { createClientSessionState } from '@/lib/chat-runtime'
import { requestGatewayForAgent } from '@/store/gateway'
import {
  $activeGatewayProfile,
  $newChatProfile,
  $newChatRoute,
  captureNewChatSource,
  selectProfile
} from '@/store/profile'
import {
  $connection,
  $currentCwd,
  applyConfiguredDefaultProjectDir,
  setActiveSessionId,
  setConnection,
  syncConfiguredDefaultProjectDir
} from '@/store/session'
import { $sessionTiles } from '@/store/session-states'
import { clearWebappSession, seedWebappSession } from '@/test/webapp-session'

import { useSessionActions } from './use-session-actions'

// A Webapp tab keeps each profile's default folder under its own key. The
// window also caches ONE folder, filled at boot for the profile it booted on,
// and a profile picked from the rail never refills it. These cases drive the
// real browser bridge, the real rail switch and the real session actions; only
// the gateway sockets are faked.

const gateway = vi.hoisted(() => ({
  ensureGatewayForProfile: vi.fn(async (_profile: string) => undefined)
}))

vi.mock('@/store/gateway', async original => ({
  ...(await original<Record<string, unknown>>()),
  activeGatewayConnectionId: vi.fn(() => null),
  activeGatewayProfileKey: () => gateway.ensureGatewayForProfile.mock.lastCall?.[0] ?? 'alpha',
  ensureGatewayForAgent: vi.fn(async () => true),
  ensureGatewayForProfile: gateway.ensureGatewayForProfile,
  requestGatewayForAgent: vi.fn(),
  retainGatewayForAgent: vi.fn(async () => () => undefined)
}))

const FOREGROUND_CREATE_DIAL = [undefined, undefined, { spawnPriority: 'foreground' }] as const

const host = window as Window & { __HERMES_BASE_PATH__?: string; __HERMES_DASHBOARD_PROFILE__?: string }

const created = (id: string) => ({ info: {}, session_id: `${id}-runtime`, stored_session_id: `${id}-stored` })

function mountActions() {
  const ref = <T>(current: T) => ({ current })
  const state = createClientSessionState()

  const requestGateway = vi.fn(
    async (method: string, _params?: Record<string, unknown>) =>
      (method === 'session.create' ? created('tab') : {}) as never
  )

  const result = renderHook(() =>
    useSessionActions({
      activeSessionId: null,
      activeSessionIdRef: ref<string | null>(null),
      busyRef: ref(false),
      creatingSessionRef: ref(false),
      ensureSessionState: () => state,
      getRouteToken: () => 'route',
      getRoutedStoredSessionId: () => null,
      navigate: vi.fn(),
      requestGateway,
      resetViewSync: vi.fn(),
      routedSessionId: null,
      runtimeIdByStoredSessionIdRef: ref(new Map()),
      selectedStoredSessionId: null,
      selectedStoredSessionIdRef: ref<string | null>(null),
      sessionStateByRuntimeIdRef: ref(new Map()),
      syncSessionStateToView: vi.fn(),
      updateSessionState: () => state
    })
  )

  return { ...result, requestGateway }
}

// Settings saves through the bridge for whichever profile the tab is on.
async function saveFolderOn(profile: string, dir: string) {
  const booted = $connection.get()
  setConnection(await window.hermesDesktop.getConnection(profile))
  await window.hermesDesktop.settings.setDefaultProjectDir(dir)
  setConnection(booted)
}

beforeEach(async () => {
  // A gateway launched as `alpha`, opened at its plain address: the tab boots
  // on the gateway's own profile with no `?profile=`, as adoptPrimaryProfile
  // leaves it, and the boot sync fills the window's folder cache.
  host.__HERMES_BASE_PATH__ = '/hermes'
  host.__HERMES_DASHBOARD_PROFILE__ = 'alpha'
  seedWebappSession('served-token')
  window.history.replaceState(null, '', '/hermes#/')
  expect(installBrowserDesktopBridge()).toBe(true)
  setConnection(await window.hermesDesktop.getConnection())
  $activeGatewayProfile.set('alpha')
  await window.hermesDesktop.settings.setDefaultProjectDir('/srv/alpha')
  await syncConfiguredDefaultProjectDir()
})

afterEach(() => {
  cleanup()
  Reflect.deleteProperty(window, 'hermesDesktop')
  delete host.__HERMES_BASE_PATH__
  delete host.__HERMES_DASHBOARD_PROFILE__
  clearWebappSession()
  document.documentElement.removeAttribute('data-hermes-desktop-host')
  window.localStorage.clear()
  window.history.replaceState(null, '', '/')
  applyConfiguredDefaultProjectDir(null)
  setActiveSessionId(null)
  setConnection(null)
  $sessionTiles.set([])
  $activeGatewayProfile.set('default')
  $newChatProfile.set(null)
  $newChatRoute.set(null)
  captureNewChatSource(null)
  vi.clearAllMocks()
})

describe('browser default folder follows the profile a new session belongs to', () => {
  it('uses the folder Settings saved on the gateway own profile', () => {
    const { result } = mountActions()

    act(() => result.current.startFreshSessionDraft())

    expect($currentCwd.get()).toBe('/srv/alpha')
  })

  it.each([
    { expected: '/srv/beta', pick: 'beta', saved: '/srv/beta' },
    { expected: '', pick: 'beta', saved: null },
    // The booted profile's folder is not the `default` profile's.
    { expected: '', pick: 'default', saved: null }
  ])(
    'after a rail switch to $pick (folder $saved), a new chat and a new tab start in $expected',
    async ({ expected, pick, saved }) => {
      if (saved) {
        await saveFolderOn(pick, saved)
      }

      const { result, requestGateway } = mountActions()

      selectProfile(pick)
      await vi.waitFor(() => expect(new URLSearchParams(window.location.search).get('profile')).toBe(pick))
      expect($connection.get()?.profile).toBe(pick)

      act(() => result.current.startFreshSessionDraft())
      expect($currentCwd.get()).toBe(expected)

      await act(() => result.current.openNewSessionTile('center', { listed: false }))

      const params = requestGateway.mock.calls.find(([method]) => method === 'session.create')?.[1]
      expect(params).toMatchObject({ profile: pick })
      expect(params?.cwd ?? '').toBe(expected)
    }
  )

  it('starts a Bot Mode "+" tab in the bot profile folder, not the active profile one', async () => {
    await saveFolderOn('writer', '/srv/writer')
    vi.mocked(requestGatewayForAgent).mockResolvedValue(created('bot'))
    const { result } = mountActions()
    // The shape the tab-strip "+" sends in Bot Mode: a route, no profile.
    const route = { connectionId: 'local', profile: 'writer' }

    await act(() =>
      result.current.openNewSessionTile('center', {
        listed: false,
        route,
        workspaceScope: { ownerRoute: route, workspaceMode: 'bots', workspaceOwnerKey: 'bot:local::writer' }
      })
    )

    expect(requestGatewayForAgent).toHaveBeenCalledWith(
      'local',
      'writer',
      'session.create',
      expect.objectContaining({ cwd: '/srv/writer', profile: 'writer' }),
      ...FOREGROUND_CREATE_DIAL
    )
  })
})

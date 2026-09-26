import { act, cleanup, render, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useHermesConfig } from '@/app/session/hooks/use-hermes-config'
import type { HermesApiRequest, HermesConnection } from '@/global'
import { setApiRequestProfile } from '@/hermes'
import { installBrowserDesktopBridge } from '@/lib/browser-desktop-bridge'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection } from '@/store/session'

import { deferred } from '../test/deferred'

import { __resetBackendSkinSync } from './backend-sync'
import { modePref, skinPref, ThemeProvider, useTheme } from './context'
import { $profileAppearance } from './profile-appearance'

// Each test uses its own profile names: a profile's one-time local seed is
// per renderer session, like the app's.

type BridgeWindow = Window & { __HERMES_SESSION_TOKEN__?: string }

let ctx: ReturnType<typeof useTheme>

function Probe() {
  ctx = useTheme()

  return null
}

function mountApp() {
  render(
    <ThemeProvider>
      <Probe />
    </ThemeProvider>
  )

  const { result } = renderHook(() => useHermesConfig({ activeSessionIdRef: { current: null } }))
  const load = () => result.current.refreshHermesConfig()

  return { load, refresh: () => act(load) }
}

const CONNECTION: HermesConnection = {
  baseUrl: 'http://127.0.0.1:9119',
  isFullscreen: false,
  logs: [],
  nativeOverlayWidth: 0,
  token: '',
  windowButtonPosition: null,
  wsUrl: ''
}

function onProfile(profile: string, connection: HermesConnection = CONNECTION) {
  act(() => {
    $activeGatewayProfile.set(profile)
    $connection.set({ ...connection, profile })
  })
}

const appearance = (theme: string, theme_mode: string) => ({ desktop: { theme, theme_mode } })

describe('profile appearance ↔ config.yaml', () => {
  // Each profile's config.yaml as the backend serves it (PUTs are recorded, not applied).
  let configs: Record<string, unknown> = {}
  let heldRead: null | ReturnType<typeof deferred<unknown>> = null

  const api = vi.fn(async (request: HermesApiRequest): Promise<unknown> => {
    if (request.method === 'PUT') {
      return { ok: true }
    }

    if (request.path !== '/api/config') {
      return {}
    }

    const held = heldRead
    heldRead = null

    return held ? held.promise : configs[request.profile ?? 'default']
  })

  const writes = () =>
    api.mock.calls
      .map(([request]) => request)
      .filter(request => request.method === 'PUT')
      .map(request => [request.path, request.profile, request.body])

  beforeEach(() => {
    window.localStorage.clear()
    __resetBackendSkinSync()
    $profileAppearance.set(null)
    configs = {}
    api.mockClear()
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api } })
  })

  afterEach(() => {
    cleanup()
    Reflect.deleteProperty(window, 'hermesDesktop')
    $connection.set(null)
    $activeGatewayProfile.set('default')
  })

  it('writes a pick as a sparse PUT for the live profile, never a preview, and a load that raced it cannot snap it back', async () => {
    configs.work = appearance('ember', 'light')
    onProfile('work')
    const { load, refresh } = mountApp()
    const raced = deferred<unknown>()
    heldRead = raced
    const racedLoad = load()

    act(() => ctx.previewTheme('everforest', 'dark'))
    expect(writes()).toEqual([])

    await act(async () => {
      ctx.setTheme('mono')
      ctx.setMode('dark')
    })

    expect(writes()).toEqual([
      ['/api/config', 'work', { config: { desktop: { theme: 'mono' } } }],
      ['/api/config', 'work', { config: { desktop: { theme_mode: 'dark' } } }]
    ])

    // That load's GET was served before the writes landed.
    await act(async () => {
      raced.resolve(appearance('ember', 'light'))
      await racedLoad
    })
    expect([ctx.themeName, ctx.mode]).toEqual(['mono', 'dark'])

    // One that began after them is the backend's word again (another client's pick).
    await refresh()
    expect([ctx.themeName, ctx.mode]).toEqual(['ember', 'light'])
  })

  it("seeds an unset config once from the profile's own local pick, never an inherited one, and adopts a set one", async () => {
    // A pre-sync install: default's picks are the legacy global keys every
    // unassigned named profile inherits; 'studio' has only its own skin.
    skinPref.assign('default', 'ember')
    modePref.assign('default', 'dark')
    skinPref.assign('studio', 'catppuccin')
    skinPref.assign('lab', 'mono')
    configs = {
      default: appearance('', ''),
      lab: appearance('everforest', ''),
      scratch: appearance('', ''),
      studio: appearance('', '')
    }

    onProfile('studio')
    const { refresh } = mountApp()
    await refresh()
    await refresh()

    for (const profile of ['default', 'scratch', 'lab']) {
      onProfile(profile)
      await refresh()
    }

    expect(writes()).toEqual([
      ['/api/config', 'studio', { config: { desktop: { theme: 'catppuccin' } } }],
      ['/api/config', 'default', { config: { desktop: { theme: 'ember', theme_mode: 'dark' } } }]
    ])
    expect([ctx.themeName, skinPref.stored('lab')]).toEqual(['everforest', 'everforest'])
  })
})

describe('Webapp profile appearance', () => {
  const configs: Record<string, unknown> = { b: appearance('everforest', 'dark'), c: appearance('ember', 'light') }

  const fetchMock = vi.fn(async (url: URL, init?: RequestInit) => {
    const body = url.pathname === '/api/config' ? configs[url.searchParams.get('profile') ?? ''] : {}

    return new Response(JSON.stringify((init?.method ?? 'GET') === 'GET' ? body : { ok: true }), { status: 200 })
  })

  beforeEach(() => {
    window.localStorage.clear()
    __resetBackendSkinSync()
    $profileAppearance.set(null)
    ;(window as BridgeWindow).__HERMES_SESSION_TOKEN__ = 'served-token'
    vi.stubGlobal('fetch', fetchMock)
  })

  afterEach(() => {
    cleanup()
    delete (window as BridgeWindow).__HERMES_SESSION_TOKEN__
    Reflect.deleteProperty(window, 'hermesDesktop')
    vi.unstubAllGlobals()
    $connection.set(null)
    $activeGatewayProfile.set('default')
  })

  // The browser bridge has no local display.skin, and this origin's cache is
  // stale or empty: the profile's config is the only place a pick made in
  // native Desktop can come from.
  it("paints the served profile's config appearance over this origin's cache, without writing it back", async () => {
    expect(installBrowserDesktopBridge()).toBe(true)
    skinPref.assign('b', 'mono')
    modePref.assign('b', 'light')
    onProfile('b', await window.hermesDesktop.getConnection('b'))
    const { refresh } = mountApp()

    await refresh()

    expect([ctx.themeName, ctx.mode]).toEqual(['everforest', 'dark'])
    expect(window.document.documentElement.dataset.hermesTheme).toBe('everforest')
    // The cache follows, so the next boot paints it before any fetch.
    expect([skinPref.stored('b'), modePref.stored('b')]).toEqual(['everforest', 'dark'])
    expect(fetchMock.mock.calls.filter(([, init]) => init?.method === 'PUT')).toEqual([])

    // A read served for another profile (the scope moved mid-flight) never paints this one.
    setApiRequestProfile('c')
    await refresh()
    expect([ctx.themeName, ctx.mode]).toEqual(['everforest', 'dark'])
  })
})

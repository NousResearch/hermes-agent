/** Foreground ownership through the real provider, config hook and API scopes. */
import { act, cleanup, render, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestLocalMode, setApiRequestProfile } from '@/api/client'
import { useHermesConfig } from '@/app/session/hooks/use-hermes-config'
import type { HermesApiRequest } from '@/global'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection } from '@/store/session'

import { deferred } from '../test/deferred'

import { __resetBackendSkinSync } from './backend-sync'
import { modePref, skinPref, ThemeProvider, useTheme } from './context'
import { $profileAppearance } from './profile-appearance'

interface Held {
  request: HermesApiRequest
  settle: ReturnType<typeof deferred<{ ok: boolean }>>
}
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

  return (force = false) => result.current.refreshHermesConfig(force)
}

const appearance = (theme: string, theme_mode: string) => ({
  desktop: { theme, theme_mode }
})

function on(connectionId: string | null, profile: string) {
  act(() => {
    setApiRequestLocalMode(connectionId === null || connectionId === 'local')
    setApiRequestConnection(connectionId)
    setApiRequestProfile(profile)
    $activeGatewayProfile.set(profile)
    $connection.set({
      baseUrl: `http://${connectionId ?? 'local'}.invalid`,
      connectionId: connectionId ?? undefined,
      profile,
      mode: connectionId === null || connectionId === 'local' ? 'local' : 'remote',
      isFullscreen: false,
      logs: [],
      nativeOverlayWidth: 0,
      token: '',
      wsUrl: '',
      windowButtonPosition: null
    })
  })
}

const state = (profile: string) => ({
  theme: ctx.themeName,
  mode: ctx.mode,
  painted: window.document.documentElement.dataset.hermesTheme,
  cachedTheme: skinPref.own(profile),
  cachedMode: modePref.own(profile)
})

// Transport has no real network and deliberately holds only requests the test chooses.
// All routing, clock/queue bookkeeping, config publication, React adoption and CSS paint are real.
describe('foreground profile appearance ownership', () => {
  let configs: Record<string, unknown>
  let held: Held[]
  let holdWrites: boolean
  let nextRead: ReturnType<typeof deferred<unknown>> | null
  let serial = 0

  const api = vi.fn(async (request: HermesApiRequest): Promise<unknown> => {
    if (request.method === 'PUT') {
      if (!holdWrites) {
        return { ok: true }
      }

      const settle = deferred<{ ok: boolean }>()
      held.push({ request, settle })

      return settle.promise
    }

    if (request.path !== '/api/config') {
      return {}
    }

    const read = nextRead
    nextRead = null

    return read ? read.promise : configs[request.connectionId ?? 'untagged']
  })

  beforeEach(() => {
    cleanup()
    window.localStorage.clear()
    __resetBackendSkinSync()
    $profileAppearance.set(null)
    configs = {}
    held = []
    holdWrites = false
    nextRead = null
    api.mockClear()
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { api }
    })
  })
  afterEach(async () => {
    await act(async () => {
      for (const entry of held) {
        entry.settle.resolve({ ok: true })
      }
    })
    cleanup()
    Reflect.deleteProperty(window, 'hermesDesktop')
    $connection.set(null)
    $activeGatewayProfile.set('default')
    setApiRequestConnection(null)
    setApiRequestLocalMode(false)
    setApiRequestProfile(null)
  })

  const owners = [
    ['A', 'B'],
    ['B', 'A'],
    ['local', 'B'],
    ['A', 'local'],
    [null, 'B']
  ] as const

  it.each(['theme', 'theme_mode'] as const)('retains rollback when a peer changes another profile %s', async field => {
    const profile = `unrelated-peer-${++serial}`
    configs.A = appearance('ember', 'light')
    on('A', profile)
    const load = mountApp()
    await act(() => load())
    holdWrites = true
    await act(async () => {
      if (field === 'theme') { ctx.setTheme('mono') } else { ctx.setMode('dark') }
    })
    expect(held).toHaveLength(1)
    const key = field === 'theme' ? 'hermes-desktop-profile-themes-v1' : 'hermes-desktop-profile-modes-v1'
    const oldValue = window.localStorage.getItem(key)
    act(() => {
      if (field === 'theme') { skinPref.put('other-profile', 'everforest') } else { modePref.put('other-profile', 'system') }
      window.dispatchEvent(new StorageEvent('storage', { key, oldValue, newValue: window.localStorage.getItem(key) }))
    })
    await act(async () => { held[0].settle.resolve({ ok: false }) })
    expect(state(profile)).toMatchObject({ theme: 'ember', cachedTheme: 'ember', mode: 'light', cachedMode: 'light' })
  })

  it.each(['theme', 'theme_mode'] as const)('keeps an inactive profile peer %s pick after an older failure', async field => {
    const profile = `inactive-peer-${++serial}`
    configs.A = appearance('ember', 'light')
    on('A', profile)
    const load = mountApp()
    await act(() => load())
    holdWrites = true
    await act(async () => {
      if (field === 'theme') { ctx.setTheme('mono') } else { ctx.setMode('dark') }
    })
    expect(held).toHaveLength(1)
    on('B', 'other-profile')
    const key = field === 'theme' ? 'hermes-desktop-profile-themes-v1' : 'hermes-desktop-profile-modes-v1'
    const oldValue = window.localStorage.getItem(key)
    act(() => {
      if (field === 'theme') { skinPref.put(profile, 'everforest') } else { modePref.put(profile, 'system') }
      window.dispatchEvent(new StorageEvent('storage', { key, oldValue, newValue: window.localStorage.getItem(key) }))
    })
    const foreground = state('other-profile')
    await act(async () => { held[0].settle.resolve({ ok: false }) })
    expect(field === 'theme' ? skinPref.own(profile) : modePref.own(profile)).toBe(field === 'theme' ? 'everforest' : 'system')
    expect(state('other-profile')).toEqual(foreground)
  })

  it.each(['theme', 'theme_mode'] as const)('keeps a peer %s pick when an older local write fails', async field => {
    const profile = `peer-${++serial}`
    configs.A = appearance('ember', 'light')
    on('A', profile)
    const load = mountApp()
    await act(() => load())
    holdWrites = true
    await act(async () => {
      if (field === 'theme') { ctx.setTheme('mono') } else { ctx.setMode('dark') }
    })
    expect(held).toHaveLength(1)
    const key = field === 'theme' ? 'hermes-desktop-profile-themes-v1' : 'hermes-desktop-profile-modes-v1'
    const oldValue = window.localStorage.getItem(key)
    act(() => {
      if (field === 'theme') { skinPref.put(profile, 'everforest') } else { modePref.put(profile, 'system') }
      window.dispatchEvent(new StorageEvent('storage', { key, oldValue, newValue: window.localStorage.getItem(key) }))
    })
    const peer = state(profile)
    await act(async () => { held[0].settle.resolve({ ok: false }) })
    expect(state(profile)).toEqual(peer)
  })

  it.each([false, true])('restores the last durable pick after queued failures (first succeeds=%s)', async firstOk => {
    const profile = `queued-${++serial}`
    configs.A = appearance('ember', 'light')
    on('A', profile)
    const load = mountApp()
    await act(() => load())
    holdWrites = true
    await act(async () => {
      ctx.setTheme('mono')
      ctx.setTheme('everforest')
    })
    expect(held).toHaveLength(1)
    await act(async () => { held[0].settle.resolve({ ok: firstOk }) })
    expect(held).toHaveLength(2)
    await act(async () => { held[1].settle.resolve({ ok: false }) })
    const expected = firstOk ? 'mono' : 'ember'
    expect(state(profile)).toMatchObject({ theme: expected, cachedTheme: expected, painted: expected })
  })

  const routes = owners.flatMap(([from, to]) => [false, true].map(named => ({ from, to, named })))

  const rollbacks = routes.flatMap(route =>
    (['theme', 'same-theme', 'mode'] as const).flatMap(field =>
      [false, true].map(returnToOwner => ({
        ...route,
        field,
        returnToOwner
      }))
    )
  )

  it.each(rollbacks)(
    'keeps $to cache after a failed $from $field write (named=$named, return=$returnToOwner)',
    async ({ from, to, named, field, returnToOwner }) => {
      const profile = named ? `rollback-${++serial}` : 'default'
      configs[from ?? 'untagged'] = appearance('ember', 'light')
      configs[to] = appearance(field === 'same-theme' ? 'mono' : 'everforest', 'dark')
      on(from, profile)
      const load = mountApp()
      await act(() => load())
      holdWrites = true
      await act(async () => {
        if (field === 'mode') {
          ctx.setMode('dark')
        } else {
          ctx.setTheme('mono')
        }
      })
      expect(held).toHaveLength(1)
      expect(held[0].request.connectionId ?? null).toBe(from)
      on(to, profile)
      await act(() => load())
      const adopted = state(profile)
      expect(adopted).toMatchObject({
        theme: field === 'same-theme' ? 'mono' : 'everforest',
        mode: 'dark'
      })

      if (returnToOwner) {
        on(from, profile)
      }

      await act(async () => held.shift()!.settle.resolve({ ok: false }))
      expect(state(profile)).toEqual(adopted)

      // A later authoritative refresh on the original owner still works.
      on(from, profile)
      await act(() => load())
      expect(state(profile)).toMatchObject({
        theme: 'ember',
        mode: 'light',
        cachedTheme: 'ember',
        cachedMode: 'light'
      })
    }
  )

  it.each(routes)(
    'keeps a new $to pick after $from fails, then rolls back its own failure (named=$named)',
    async ({ from, to, named }) => {
      const profile = named ? `new-pick-${++serial}` : 'default'
      configs[from ?? 'untagged'] = appearance('ember', 'light')
      configs[to] = appearance('everforest', 'dark')
      on(from, profile)
      const load = mountApp()
      await act(() => load())
      holdWrites = true
      await act(async () => ctx.setTheme('mono'))
      on(to, profile)
      await act(() => load())
      const adopted = state(profile)
      await act(async () => ctx.setTheme('mono'))
      expect(held).toHaveLength(2)
      const picked = state(profile)
      expect(picked).toMatchObject({ theme: 'mono', mode: 'dark', cachedTheme: 'mono' })
      await act(async () => held.shift()!.settle.resolve({ ok: false }))
      expect(state(profile)).toEqual(picked)
      await act(async () => held.shift()!.settle.resolve({ ok: false }))
      expect(state(profile)).toEqual(adopted)
    }
  )

  it.each(routes)(
    'keeps $to publication and cache when a late $from read settles (named=$named)',
    async ({ from, to, named }) => {
      const profile = named ? `read-${++serial}` : 'default'
      configs[from ?? 'untagged'] = appearance('ember', 'light')
      configs[to] = appearance('everforest', 'dark')
      on(from, profile)
      const load = mountApp()
      await act(() => load())
      const delayed = deferred<unknown>()
      nextRead = delayed
      let oldLoad: Promise<void>
      act(() => {
        oldLoad = load()
      })
      on(to, profile)
      await act(() => load())
      const adopted = state(profile)
      const publication = $profileAppearance.get()
      expect(publication?.owner).toBe(`${to}::${profile}`)
      expect(adopted).toMatchObject({
        theme: 'everforest',
        mode: 'dark',
        painted: 'everforest'
      })
      await act(async () => {
        delayed.resolve(appearance('ember', 'light'))
        await oldLoad!
      })
      expect($profileAppearance.get()).toBe(publication)
      expect(state(profile)).toEqual(adopted)

      on(from, profile)
      await act(() => load())
      expect(state(profile)).toMatchObject({
        theme: 'ember',
        mode: 'light',
        cachedTheme: 'ember',
        cachedMode: 'light'
      })
    }
  )
})

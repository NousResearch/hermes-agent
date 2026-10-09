/** Revision ordering through the real provider, config hook and API scopes. */
import { act, cleanup, render, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestLocalMode, setApiRequestProfile } from '@/api/client'
import { useHermesConfig } from '@/app/session/hooks/use-hermes-config'
import type { HermesApiRequest } from '@/global'
import { $notifications, clearNotifications } from '@/store/notifications'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection } from '@/store/session'
import { dropTilesForProfile, migrateTilesForProfile } from '@/store/session-states'

import { deferred } from '../test/deferred'

import { __resetBackendSkinSync } from './backend-sync'
import { modePref, skinPref, type ThemeMode, ThemeProvider, useTheme } from './context'
import { $profileAppearance } from './profile-appearance'

type Field = 'theme' | 'theme_mode'

/** One profile's config.yaml as hermes serves it: every save that lands advances its revision. */
interface Disk {
  desktop: Record<Field, string>
  revision: number
}

interface HeldWrite {
  request: HermesApiRequest
  /** Apply the save to config.yaml now; it answers later. */
  land: () => number
  answer: (ok?: boolean) => void
}

let ctx: ReturnType<typeof useTheme>

function Probe() {
  ctx = useTheme()

  return null
}

const FIELDS = {
  theme: { pref: skinPref, values: ['ember', 'mono', 'everforest'] },
  theme_mode: { pref: modePref, values: ['light', 'dark', 'system'] }
} as const

/** Another window's localStorage writes reach this one only as storage events. */
function fromPeer(write: () => void) {
  const snapshot = () => {
    const keys = Array.from({ length: window.localStorage.length }, (_, index) => window.localStorage.key(index) ?? '')

    return new Map(keys.map(key => [key, window.localStorage.getItem(key)]))
  }

  const before = snapshot()
  write()
  const after = snapshot()

  act(() => {
    for (const key of new Set([...before.keys(), ...after.keys()])) {
      const [oldValue, newValue] = [before.get(key) ?? null, after.get(key) ?? null]

      if (oldValue !== newValue) {
        window.dispatchEvent(new StorageEvent('storage', { key, oldValue, newValue }))
      }
    }
  })
}

/** Let every request a step started answer, and what those answers start. */
const drain = () => act(() => new Promise(resolve => setTimeout(resolve, 0)))

describe('profile appearance ordered by config revision', () => {
  let disks: Record<string, Disk>
  let heldWrites: HeldWrite[]
  let holdWrites: boolean
  let heldReads: (() => void)[]
  let holdNextRead: boolean
  let serial = 0

  const diskOf = (request: HermesApiRequest) => disks[`${request.connectionId ?? 'untagged'}::${request.profile}`]

  const land = (disk: Disk, patch: Partial<Record<Field, string>>) => {
    Object.assign(disk.desktop, patch)

    return ++disk.revision
  }

  const api = vi.fn(async (request: HermesApiRequest): Promise<unknown> => {
    const disk = diskOf(request)

    if (request.method === 'PUT') {
      const patch = (request.body as { config: { desktop: Partial<Record<Field, string>> } }).config.desktop

      if (!holdWrites) {
        return { ok: true, revision: land(disk, patch) }
      }

      const answered = deferred<unknown>()
      let revision: number | undefined

      heldWrites.push({
        request,
        land: () => (revision = land(disk, patch)),
        answer: (ok = true) =>
          answered.resolve(ok ? { ok: true, revision: revision ?? land(disk, patch) } : { ok: false })
      })

      return answered.promise
    }

    if (request.path.split('?')[0] !== '/api/config') {
      return {}
    }

    const config = { desktop: { ...disk.desktop } }
    const served = request.path.includes('with_revision=true') ? { config, revision: disk.revision } : config

    if (!holdNextRead) {
      return served
    }

    // Served now, answered whenever the test says.
    holdNextRead = false
    const answered = deferred<unknown>()
    heldReads.push(() => answered.resolve(served))

    return answered.promise
  })

  function on(connectionId: string, profile: string) {
    act(() => {
      setApiRequestLocalMode(connectionId === 'local')
      setApiRequestConnection(connectionId)
      setApiRequestProfile(profile)
      $activeGatewayProfile.set(profile)
      $connection.set({
        baseUrl: `http://${connectionId}.invalid`,
        connectionId,
        profile,
        mode: connectionId === 'local' ? 'local' : 'remote',
        isFullscreen: false,
        logs: [],
        nativeOverlayWidth: 0,
        token: '',
        wsUrl: '',
        windowButtonPosition: null
      })
    })
  }

  /** This window, loaded on `connectionId::profile` whose config.yaml is at `revision`. */
  async function windowOn(connectionId: string, revision = 1, profile = `p-${++serial}`) {
    disks[`${connectionId}::${profile}`] = { desktop: { theme: 'ember', theme_mode: 'light' }, revision }
    on(connectionId, profile)
    render(
      <ThemeProvider>
        <Probe />
      </ThemeProvider>
    )
    const { result } = renderHook(() => useHermesConfig({ activeSessionIdRef: { current: null } }))
    const refresh = () => result.current.refreshHermesConfig()
    const load = () => act(refresh)
    await load()

    return { load, profile, refresh }
  }

  /** Start a config load that is served now and answers when the test says. */
  function staleLoad(refresh: () => Promise<void>) {
    holdNextRead = true
    let loading!: Promise<void>
    act(() => {
      loading = refresh()
    })

    return () =>
      act(async () => {
        heldReads.shift()!()
        await loading
      })
  }

  /** A peer window's save: it lands on the backend, then that window caches it (storage events here). */
  function peerSaves(connectionId: string, profile: string, field: Field, value: string) {
    land(disks[`${connectionId}::${profile}`], { [field]: value })
    fromPeer(() => FIELDS[field].pref.pick(profile, value))
  }

  const pick = (field: Field, value: string) =>
    act(async () => (field === 'theme' ? ctx.setTheme(value) : ctx.setMode(value as ThemeMode)))

  const shown = (profile: string, field: Field) => ({
    view: field === 'theme' ? ctx.themeName : ctx.mode,
    cache: FIELDS[field].pref.own(profile),
    ...(field === 'theme' ? { painted: window.document.documentElement.dataset.hermesTheme } : {})
  })

  const showing = (field: Field, value: string) => ({
    view: value,
    cache: value,
    ...(field === 'theme' ? { painted: value } : {})
  })

  const errors = () => $notifications.get().filter(notice => notice.kind === 'error')

  beforeEach(() => {
    cleanup()
    window.localStorage.clear()
    clearNotifications()
    __resetBackendSkinSync()
    $profileAppearance.set(null)
    disks = {}
    heldWrites = []
    holdWrites = false
    heldReads = []
    holdNextRead = false
    api.mockClear()
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api } })
  })

  afterEach(async () => {
    await act(async () => heldWrites.forEach(write => write.answer()))
    cleanup()
    Reflect.deleteProperty(window, 'hermesDesktop')
    $connection.set(null)
    $activeGatewayProfile.set('default')
    setApiRequestConnection(null)
    setApiRequestLocalMode(false)
    setApiRequestProfile(null)
  })

  it.each(['theme', 'theme_mode'] as const)(
    'keeps the newest %s whatever order racing reads answer in',
    async field => {
      const { load, profile, refresh } = await windowOn('A')
      const [, theirs] = FIELDS[field].values
      const answerOld = staleLoad(refresh)
      // Another origin (the Webapp, another backend) saves: nothing tells this window but its reads.
      land(disks[`A::${profile}`], { [field]: theirs })
      await load()
      expect(shown(profile, field)).toEqual(showing(field, theirs))

      await answerOld()
      expect(shown(profile, field)).toEqual(showing(field, theirs))
    }
  )

  it.each(['theme', 'theme_mode'] as const)(
    'never lets a read served before this window saved its %s snap the pick back',
    async field => {
      const { profile, refresh } = await windowOn('A')
      const [, mine] = FIELDS[field].values
      const answerOld = staleLoad(refresh)
      await pick(field, mine)
      await answerOld()
      expect(shown(profile, field)).toEqual(showing(field, mine))
    }
  )

  // This window and a peer on the same owner pick at once. Whichever save lands
  // last is the config, in either answer order, and this window ends there.
  // (The shared cache converges through the peer's own re-read, which needs a
  // real second window: smoke-appearance-transport.mjs.)
  const races = (['theme', 'theme_mode'] as const).flatMap(field =>
    (['mine', 'theirs'] as const).flatMap(landsLast =>
      (['mine', 'theirs'] as const).map(answersLast => ({ field, landsLast, answersLast }))
    )
  )

  it.each(races)(
    'ends on the $field save that landed last ($landsLast lands last, $answersLast answers last)',
    async ({ field, landsLast, answersLast }) => {
      const { profile } = await windowOn('A')
      const [, mine, theirs] = FIELDS[field].values
      holdWrites = true
      await pick(field, mine)
      const [write] = heldWrites
      const peerLands = () => land(disks[`A::${profile}`], { [field]: theirs })

      if (landsLast === 'mine') {
        peerLands()
        write.land()
      } else {
        write.land()
        peerLands()
      }

      // The peer caches its save once it answers; that reaches this window as storage events.
      const peerAnswers = () => fromPeer(() => FIELDS[field].pref.pick(profile, theirs))

      if (answersLast === 'mine') {
        peerAnswers()
        await drain()
        await act(async () => write.answer())
      } else {
        await act(async () => write.answer())
        peerAnswers()
      }

      await drain()
      const saved = landsLast === 'mine' ? mine : theirs
      expect(disks[`A::${profile}`].desktop[field]).toBe(saved)
      expect(shown(profile, field)).toMatchObject({ view: saved, ...(field === 'theme' ? { painted: saved } : {}) })
    }
  )

  it.each(['theme', 'theme_mode'] as const)(
    'rolls a failed %s save back to the newest config, a peer save made meanwhile included, and says so once',
    async field => {
      const { profile } = await windowOn('A')
      const [, mine, theirs] = FIELDS[field].values
      holdWrites = true
      await pick(field, mine)
      peerSaves('A', profile, field, theirs)
      await drain()
      // Still painting the pick in flight.
      expect(shown(profile, field).view).toBe(mine)

      await act(async () => heldWrites.shift()!.answer(false))
      expect(shown(profile, field)).toEqual(showing(field, theirs))
      expect(errors()).toHaveLength(1)
    }
  )

  it.each(['theme', 'theme_mode'] as const)(
    "never repaints a %s saved on another gateway's profile of the same name",
    async field => {
      const { profile } = await windowOn('A')
      const [base, , theirs] = FIELDS[field].values
      disks[`B::${profile}`] = { desktop: { theme: 'ember', theme_mode: 'light' }, revision: 1 }
      peerSaves('B', profile, field, theirs)
      await drain()
      expect(shown(profile, field).view).toBe(base)
      expect(disks[`A::${profile}`].desktop[field]).toBe(base)
    }
  )

  it('orders revisions per owner: a high one on one gateway never blocks another', async () => {
    const { profile } = await windowOn('A', 1_000_000)
    disks[`B::${profile}`] = { desktop: { theme: 'everforest', theme_mode: 'dark' }, revision: 3 }
    on('B', profile)
    const { result } = renderHook(() => useHermesConfig({ activeSessionIdRef: { current: null } }))
    await act(() => result.current.refreshHermesConfig())
    expect([ctx.themeName, ctx.mode]).toEqual(['everforest', 'dark'])
  })

  it.each([
    ['deleted', (profile: string) => dropTilesForProfile(profile, { connectionId: 'local', profile })],
    ['renamed onto', (profile: string) => migrateTilesForProfile(`${profile}-old`, profile)]
  ] as const)('forgets the revisions of a profile %s, whose config.yaml may be older', async (_label, lifecycle) => {
    const { load, profile } = await windowOn('local', 500)
    // The successor's config.yaml carries an older revision (a renamed file keeps its mtime).
    disks[`local::${profile}`] = { desktop: { theme: 'mono', theme_mode: 'dark' }, revision: 7 }
    await load()
    expect(ctx.themeName).toBe('ember')

    lifecycle(profile)
    await load()
    expect([ctx.themeName, ctx.mode]).toEqual(['mono', 'dark'])
  })
})

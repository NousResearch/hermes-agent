// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { Profiler } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type * as SessionRequestRouter from '@/store/session-request-router'

import { entityKey } from './identity'
import { leaderModelIdForOwner } from './leader-runtime'
import type * as LeaderSessions from './leader-sessions'
import type { LunarCityIntent, LunarCityWorldHandle } from './model'
import { $lunarCitySnapshot, applyLunarDelta, createLunarCitySnapshot } from './store'

import { disposeLunarCityRuntime, LunarCity, lunarCityHudStatus, lunarCityManifestUrl } from './index'

const {
  applySnapshot,
  createRegisteredKanbanCitySource,
  createWorld,
  destroyWorld,
  dispatchCamera,
  emitWorldIntent,
  getCameraState,
  kanbanSource,
  leaderAnimation,
  loadManifest,
  requestForSessionProfile,
  resolveLeaderSession,
  setQuality,
  setReducedMotion,
  startReconciler,
  stopReconciler,
  worldHandle,
  worldDeferred
} = vi.hoisted(() => {
  const apply = vi.fn()
  const destroy = vi.fn()
  const dispatch = vi.fn()
  const readCameraState = vi.fn(() => ({ focusedEntityKey: undefined, following: false }))
  const readEntityCameraOrder = vi.fn(() => [] as never[])
  const setLeaderAnimation = vi.fn()
  const setQuality = vi.fn()
  const setReducedMotion = vi.fn()
  type TestWorldHandle = {
    applySnapshot: typeof apply
    destroy: typeof destroy
    dispatchCamera: typeof dispatch
    getCameraState: typeof readCameraState
    getEntityCameraOrder: typeof readEntityCameraOrder
    setLeaderAnimation: typeof setLeaderAnimation
    setQuality: typeof setQuality
    setReducedMotion: typeof setReducedMotion
  }

  const handle: TestWorldHandle = {
    applySnapshot: apply,
    destroy,
    dispatchCamera: dispatch,
    getCameraState: readCameraState,
    getEntityCameraOrder: readEntityCameraOrder,
    setLeaderAnimation,
    setQuality,
    setReducedMotion
  }

  let resolveWorld!: (handle: TestWorldHandle) => void

  const deferred = new Promise<TestWorldHandle>(resolve => {
    resolveWorld = resolve
  })

  const source = { onFrame: vi.fn(), read: vi.fn(), start: vi.fn() }
  const stopLive = vi.fn()

  let worldIntent: (intent: LunarCityIntent) => void = () => undefined

  const create = vi.fn((_canvas: unknown, _manifest: unknown, onIntent: (intent: LunarCityIntent) => void) => {
    worldIntent = onIntent

    return Promise.resolve(handle)
  })

  return {
    applySnapshot: apply,
    createRegisteredKanbanCitySource: vi.fn(() => source),
    createWorld: create,
    dispatchCamera: dispatch,
    destroyWorld: destroy,
    emitWorldIntent: (intent: LunarCityIntent) => worldIntent(intent),
    getCameraState: readCameraState,
    getEntityCameraOrder: readEntityCameraOrder,
    kanbanSource: source,
    leaderAnimation: setLeaderAnimation,
    loadManifest: vi.fn(async () => ({
      models: [],
      characterAssets: { leaders: ['owl', 'fox', 'badger', 'otter', 'bird', 'stag'].map(id => ({ id })) }
    })),
    requestForSessionProfile: vi.fn(async () => ({ status: 'queued' })),
    resolveLeaderSession: vi.fn(async () => ({ runtimeId: 'runtime-owl', storedId: 'stored-owl' })),
    setQuality,
    setReducedMotion,
    startReconciler: vi.fn(() => stopLive),
    stopReconciler: stopLive,
    worldHandle: handle,
    worldDeferred: { promise: deferred, resolve: resolveWorld }
  }
})

vi.mock('./manifest', () => ({ loadWorldManifest: loadManifest }))
vi.mock('./adapters/kanban', () => ({ createRegisteredKanbanCitySource }))
vi.mock('./adapters/reconciler', () => ({ startLunarCityReconciler: startReconciler }))
vi.mock('./world/create-world', () => ({ createLunarCityWorld: createWorld }))
vi.mock('./leader-sessions', async importOriginal => ({
  ...(await importOriginal<typeof LeaderSessions>()),
  resolveLeaderSession
}))
vi.mock('@/store/session-request-router', async importOriginal => ({
  ...(await importOriginal<typeof SessionRequestRouter>()),
  requestForSessionProfile
}))

afterEach(() => {
  cleanup()
  $lunarCitySnapshot.set(createLunarCitySnapshot())
  vi.clearAllMocks()
  vi.unstubAllGlobals()
})

function publishAccessibleEntities(): { leaderKey: never; workerKey: never } {
  const workerIdentity = {
    connectionId: 'source-a',
    kind: 'session' as const,
    profile: 'research',
    sessionId: 'pip'
  }

  const leaderIdentity = { connectionId: 'source-a', kind: 'profile' as const, profile: 'fox-scientist' }
  const workerKey = entityKey(workerIdentity)
  const leaderKey = entityKey(leaderIdentity)

  $lunarCitySnapshot.set({
    entities: new Map([
      [
        workerKey,
        {
          animation: 'work' as const,
          authority: 'authoritative' as const,
          destination: 'lab' as const,
          identity: workerIdentity,
          key: workerKey,
          observedAt: 42
        }
      ],
      [
        leaderKey,
        {
          animation: 'rest' as const,
          authority: 'authoritative' as const,
          destination: 'lab' as const,
          identity: leaderIdentity,
          key: leaderKey,
          observedAt: 42,
          presentation: {
            configuredTitle: 'Fox Scientist',
            groups: [{ id: 'research', name: 'Research Lab' }],
            metadata: { observedAt: 42, source: 'profiles:source-a', state: 'fresh' as const },
            placement: { lodHint: 0, overflow: false, primaryGroupId: 'research', slot: 0 }
          }
        }
      ]
    ]),
    observedAt: 42,
    revision: 3,
    sources: [
      { authority: 'authoritative', observedAt: 42, source: 'profiles:source-a' },
      { authority: 'authoritative', observedAt: 42, source: 'session:source-a' }
    ]
  })

  return { leaderKey: leaderKey as never, workerKey: workerKey as never }
}

describe('lunarCityManifestUrl', () => {
  it('uses a same-origin host manifest while rejecting cross-origin absolute URLs', () => {
    const locationLike = {
      href: 'https://hermes.local/city?lunarCityManifest=%2Fassets%2Fcity.json',
      origin: 'https://hermes.local',
      search: '?lunarCityManifest=%2Fassets%2Fcity.json'
    } as Location

    expect(lunarCityManifestUrl(locationLike)).toBe('/assets/city.json')
    expect(
      lunarCityManifestUrl({
        href: 'https://hermes.local/city?lunarCityManifest=https%3A%2F%2Fevil.example%2Fcity.json',
        origin: 'https://hermes.local',
        search: '?lunarCityManifest=https%3A%2F%2Fevil.example%2Fcity.json'
      } as Location)
    ).toBe('./lunar-city/v2/world-manifest.v2.json')
  })

  it('falls back to the built-in manifest for missing or malformed location input', () => {
    expect(lunarCityManifestUrl()).toBe('./lunar-city/v2/world-manifest.v2.json')
    expect(
      lunarCityManifestUrl({
        href: 'not a URL',
        origin: '',
        search: '?lunarCityManifest=%E0%A4%A'
      } as Location)
    ).toBe('./lunar-city/v2/world-manifest.v2.json')
  })
})

describe('LunarCity', () => {
  it('reports HUD health from renderer and immutable source truth', () => {
    const empty = createLunarCitySnapshot()
    expect(lunarCityHudStatus('ready', empty)).toBe('EMPTY')
    expect(lunarCityHudStatus('unavailable', empty)).toBe('RENDERER UNAVAILABLE')
    expect(lunarCityHudStatus('loading', empty)).toBe('STARTING')
    expect(lunarCityHudStatus('degraded', empty)).toBe('DEGRADED')
    const { workerKey } = publishAccessibleEntities()
    expect(workerKey).toBeTruthy()
    expect(lunarCityHudStatus('ready', $lunarCitySnapshot.get())).toBe('LIVE')
    expect(
      lunarCityHudStatus('ready', {
        ...$lunarCitySnapshot.get(),
        sources: [{ authority: 'stale', observedAt: 42, source: 'session:source-a' }]
      })
    ).toBe('STALE')
  })
  it('opens the world with one ready 3D canvas and no runtime source-art image', async () => {
    const interval = vi.spyOn(window, 'setInterval')
    render(<LunarCity onOpenMemoryGraph={vi.fn()} />)

    expect(screen.getByRole('heading', { name: 'Lunar City' })).toBeTruthy()
    const canvas = screen.getByLabelText('Interactive 3D Lunar City')
    expect(canvas.closest('.lunar-city')?.classList.contains('h-full')).toBe(true)

    await waitFor(() => expect(canvas.getAttribute('data-world-status')).toBe('ready'))
    expect(globalThis.document.querySelectorAll('canvas')).toHaveLength(1)
    expect(globalThis.document.querySelector('img[src*="moon-settlement-approved.jpg"]')).toBeNull()
    expect(screen.queryByRole('button', { name: /Open Research Lab/i })).toBeNull()
    expect(screen.queryByRole('button', { name: /Inspect Fox Scientist/i })).toBeNull()
    expect(screen.queryByText('SIMULATION')).toBeNull()
    expect(screen.queryByText('MISSIONS')).toBeNull()
    expect(interval.mock.calls.some(([, delay]) => delay === 1_200)).toBe(false)
  })

  it('opens exact worker controls from the Babylon entity pick without replacing the city canvas', async () => {
    const identity = { connectionId: 'source-b', kind: 'session', profile: 'builder', sessionId: 'session-9' } as const
    const key = entityKey(identity)
    render(<LunarCity onOpenMemoryGraph={vi.fn()} />)
    await waitFor(() => expect(screen.getByLabelText('Interactive 3D Lunar City').dataset.worldStatus).toBe('ready'))

    act(() => {
      applyLunarDelta({
        observedAt: 100,
        removals: [],
        revision: 1,
        sources: [{ authority: 'authoritative', observedAt: 100, source: 'session:source-b' }],
        upserts: [
          {
            animation: 'working',
            authority: 'authoritative',
            destination: 'project',
            identity,
            key,
            observedAt: 100
          }
        ]
      })
      emitWorldIntent({ entityKey: key, kind: 'select-focus' })
    })

    expect(screen.getByRole('region', { name: 'Lunar City worker controls' })).toBeTruthy()
    expect(screen.getByText('source-b')).toBeTruthy()
    expect(screen.getAllByLabelText('Interactive 3D Lunar City')).toHaveLength(1)
  })

  it('starts the live writer only inside the Lunar City route with the registered-owner Kanban source', async () => {
    const { unmount } = render(<LunarCity onOpenMemoryGraph={vi.fn()} />)

    await waitFor(() => expect(screen.getByLabelText('Interactive 3D Lunar City').dataset.worldStatus).toBe('ready'))

    await waitFor(() =>
      expect(createRegisteredKanbanCitySource).toHaveBeenCalledWith(
        expect.objectContaining({
          manifest: expect.objectContaining({ models: [] })
        })
      )
    )
    expect(startReconciler).toHaveBeenCalledWith({ optionalSources: [kanbanSource] })

    unmount()
    expect(stopReconciler).toHaveBeenCalledOnce()
  })

  it('applies many live snapshot publications to the ready world without recreating or rerendering the route', async () => {
    const commits = vi.fn()
    render(
      <Profiler id="lunar-city" onRender={commits}>
        <LunarCity onOpenMemoryGraph={vi.fn()} />
      </Profiler>
    )
    await waitFor(() => expect(screen.getByLabelText('Interactive 3D Lunar City').dataset.worldStatus).toBe('ready'))
    await waitFor(() => expect(startReconciler).toHaveBeenCalledOnce())
    await act(async () => {
      await Promise.resolve()
      await Promise.resolve()
    })

    const rendersBeforePublications = commits.mock.calls.length
    const applicationsBeforePublications = applySnapshot.mock.calls.length

    const publications = Array.from({ length: 8 }, (_, index) =>
      createLunarCitySnapshot({ observedAt: 800 + index, revision: 2 + index })
    )

    await act(async () => {
      for (const next of publications) {
        $lunarCitySnapshot.set(next)
      }
    })

    expect(applySnapshot.mock.calls.slice(applicationsBeforePublications).map(([next]) => next)).toEqual(publications)
    expect(createWorld).toHaveBeenCalledOnce()
    // The route and canvas remain stable. The isolated semantic entity list
    // coalesces the batch into one React commit instead of one per source publication.
    expect(commits.mock.calls.length - rendersBeforePublications).toBeLessThanOrEqual(1)
  })

  it('stops the reconciler, unsubscribes snapshots, then destroys the ready world', () => {
    const order: string[] = []
    const world = { destroy: () => order.push('world destroy') } as unknown as LunarCityWorldHandle

    disposeLunarCityRuntime(
      () => order.push('reconciler stop'),
      () => order.push('snapshot unsubscribe'),
      world
    )

    expect(order).toEqual(['reconciler stop', 'snapshot unsubscribe', 'world destroy'])
  })

  it('destroys a world that finishes creating after the route unmounts', async () => {
    createWorld.mockReturnValueOnce(worldDeferred.promise)
    const { unmount } = render(<LunarCity onOpenMemoryGraph={vi.fn()} />)

    await act(async () => {
      await Promise.resolve()
    })
    unmount()
    await act(async () => {
      worldDeferred.resolve(worldHandle)
      await worldDeferred.promise
    })

    expect(destroyWorld).toHaveBeenCalledOnce()
  })

  it('exposes the memory graph escape hatch', () => {
    const onOpenMemoryGraph = vi.fn()
    render(<LunarCity onOpenMemoryGraph={onOpenMemoryGraph} />)

    fireEvent.click(screen.getByRole('button', { name: 'Open memory graph' }))

    expect(onOpenMemoryGraph).toHaveBeenCalledOnce()
  })

  it('uses only immutable snapshot entities for its accessible world interactions', async () => {
    publishAccessibleEntities()
    render(<LunarCity onOpenMemoryGraph={vi.fn()} />)

    await waitFor(() =>
      expect(screen.getByRole('button', { name: /Session Pip.*Working.*Research Lab/i })).toBeTruthy()
    )
    expect(screen.queryByRole('button', { name: /Inspect .* worker/i })).toBeNull()
    expect(screen.queryByRole('progressbar')).toBeNull()
    expect(screen.getByTestId('lunar-city-viewport').getAttribute('data-camera')).toBe('angled-simcity')
  })

  it('uses native Babylon camera controls instead of a CSS viewport zoom', () => {
    render(<LunarCity onOpenMemoryGraph={vi.fn()} />)

    const viewport = screen.getByTestId('lunar-city-viewport')

    expect(viewport.getAttribute('data-zoom')).toBeNull()
    expect(screen.getByRole('button', { name: 'Zoom In' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Return to City' })).toBeTruthy()
  })

  it('routes the native camera controls to the existing Babylon world without recreating it', async () => {
    render(<LunarCity onOpenMemoryGraph={vi.fn()} />)

    await waitFor(() => expect(screen.getByLabelText('Interactive 3D Lunar City').dataset.worldStatus).toBe('ready'))
    fireEvent.click(screen.getByRole('button', { name: 'Rotate Left' }))
    fireEvent.click(screen.getByRole('button', { name: 'Return to City' }))

    expect(dispatchCamera).toHaveBeenCalledWith({ kind: 'orbit', deltaAlpha: -0.22, deltaBeta: 0 })
    expect(dispatchCamera).toHaveBeenCalledWith({ kind: 'return-to-city' })
    expect(createWorld).toHaveBeenCalledOnce()
  })

  it('mounts text-first entity, source, quality, and camera controls over the same world authority', async () => {
    const { workerKey } = publishAccessibleEntities()
    render(<LunarCity onOpenMemoryGraph={vi.fn()} />)

    await waitFor(() => expect(screen.getByLabelText('Interactive 3D Lunar City').dataset.worldStatus).toBe('ready'))
    const pip = screen.getByRole('button', { name: /Session Pip.*Working.*Research Lab.*Authoritative/i })

    fireEvent.click(pip)
    expect(dispatchCamera).toHaveBeenCalledWith({ entityKey: workerKey, follow: false, kind: 'focus' })
    expect(screen.getByRole('region', { name: 'Lunar City worker controls' })).toBeTruthy()
    expect(screen.getByRole('region', { name: 'Source health' }).textContent).toMatch(/Authoritative.*Healthy/i)

    const quality = screen.getByRole('combobox', { name: /3D quality/i })
    expect((quality as HTMLSelectElement).value).toBe('efficient')
    fireEvent.change(quality, { target: { value: 'balanced' } })
    expect(setQuality).toHaveBeenCalledWith('balanced')

    act(() => emitWorldIntent({ kind: 'camera-state', state: { focusedEntityKey: workerKey, following: false } }))
    expect(screen.getByRole('status', { name: 'Camera position' }).textContent).toBe('Focused on Session Pip')
    expect(screen.getByRole('status', { name: 'Camera position' }).textContent).not.toContain(String(workerKey))
  })

  it('retires a lost world immediately but waits for the matching restored context before one rebuild', async () => {
    const { workerKey } = publishAccessibleEntities()
    const attempts: Array<(handle: typeof worldHandle) => void> = []
    const contextStates: string[] = []
    let contextState = 'available'

    const replacement = {
      ...worldHandle,
      applySnapshot: vi.fn(),
      destroy: vi.fn()
    } as typeof worldHandle

    const deferredCreation = () =>
      new Promise<typeof worldHandle>(resolve => {
        contextStates.push(contextState)
        attempts.push(resolve)
      })

    createWorld.mockImplementationOnce(deferredCreation).mockImplementationOnce(deferredCreation)

    render(<LunarCity onOpenMemoryGraph={vi.fn()} />)
    const canvas = screen.getByLabelText('Interactive 3D Lunar City')
    await waitFor(() => expect(attempts).toHaveLength(1))
    await act(async () => attempts.shift()!(worldHandle))
    await waitFor(() => expect(canvas.dataset.worldStatus).toBe('ready'))

    const latest = { ...$lunarCitySnapshot.get(), observedAt: 900, revision: 9 }
    act(() => $lunarCitySnapshot.set(latest))
    const lost = new Event('webglcontextlost', { cancelable: true })
    contextState = 'lost'
    fireEvent(canvas, lost)

    expect(lost.defaultPrevented).toBe(true)
    expect(destroyWorld).toHaveBeenCalledOnce()
    expect(createWorld).toHaveBeenCalledOnce()
    expect(canvas.dataset.worldStatus).toBe('restoring')

    contextState = 'restored'
    fireEvent(canvas, new Event('webglcontextrestored'))
    await waitFor(() => expect(attempts).toHaveLength(1))
    expect(createWorld).toHaveBeenCalledTimes(2)
    fireEvent(canvas, new Event('webglcontextrestored'))
    expect(createWorld).toHaveBeenCalledTimes(2)
    await act(async () => attempts.shift()!(replacement))
    await waitFor(() => expect(canvas.dataset.worldStatus).toBe('ready'))
    expect(replacement.applySnapshot).toHaveBeenLastCalledWith(latest)
    expect(contextStates).toEqual(['available', 'restored'])
    expect(startReconciler).toHaveBeenCalledOnce()

    contextState = 'lost'
    fireEvent(canvas, new Event('webglcontextlost', { cancelable: true }))
    await act(async () => Promise.resolve())
    expect(createWorld).toHaveBeenCalledTimes(2)
    expect(
      screen.getByRole('button', { name: new RegExp(String(workerKey).replace(/[.*+?^${}()|[\]\\]/gu, '\\$&')) })
    ).toBeTruthy()
  })

  it('keeps exact leader chat, entities, quality, and authorized controls usable when restoration fails', async () => {
    publishAccessibleEntities()
    render(<LunarCity onOpenMemoryGraph={vi.fn()} />)
    const canvas = screen.getByLabelText('Interactive 3D Lunar City')
    await waitFor(() => expect(canvas.dataset.worldStatus).toBe('ready'))
    createWorld.mockRejectedValueOnce(new Error('replacement engine unavailable'))

    fireEvent(canvas, new Event('webglcontextlost', { cancelable: true }))
    expect(createWorld).toHaveBeenCalledOnce()
    expect(canvas.dataset.worldStatus).toBe('restoring')
    fireEvent(canvas, new Event('webglcontextrestored'))

    await waitFor(() => expect(screen.getByText(/3D world renderer unavailable/i)).toBeTruthy())
    expect(screen.getByRole('button', { name: /Session Pip.*Working.*Research Lab/i })).toBeTruthy()
    expect(screen.getByRole('combobox', { name: /3D quality/i })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Talk to fox-scientist leader on source-a' }))
    await waitFor(() => expect(screen.getByRole('dialog', { name: 'fox-scientist leader conversation' })).toBeTruthy())
    fireEvent.click(screen.getByRole('button', { name: /Session Pip.*Working.*Research Lab/i }))
    expect(screen.getByRole('region', { name: 'Lunar City worker controls' })).toBeTruthy()
    expect(createWorld).toHaveBeenCalledTimes(2)
  })

  it('destroys a late replacement and publishes nothing after route teardown', async () => {
    publishAccessibleEntities()
    let resolveReplacement!: (handle: LunarCityWorldHandle) => void

    const replacementPromise = new Promise<LunarCityWorldHandle>(resolve => {
      resolveReplacement = resolve
    })

    const replacement = {
      ...worldHandle,
      applySnapshot: vi.fn(),
      destroy: vi.fn()
    } as unknown as LunarCityWorldHandle

    const { unmount } = render(<LunarCity onOpenMemoryGraph={vi.fn()} />)
    const canvas = screen.getByLabelText('Interactive 3D Lunar City')
    await waitFor(() => expect(canvas.dataset.worldStatus).toBe('ready'))
    createWorld.mockReturnValueOnce(replacementPromise as never)

    fireEvent(canvas, new Event('webglcontextlost', { cancelable: true }))
    fireEvent(canvas, new Event('webglcontextrestored'))
    unmount()
    resolveReplacement(replacement)
    await act(async () => replacementPromise)

    expect(replacement.destroy).toHaveBeenCalledOnce()
    expect(replacement.applySnapshot).not.toHaveBeenCalled()
    expect(stopReconciler).toHaveBeenCalledOnce()
  })

  it('ignores a late context-restored publication after route teardown', async () => {
    render(<LunarCity onOpenMemoryGraph={vi.fn()} />)
    const canvas = screen.getByLabelText('Interactive 3D Lunar City')
    await waitFor(() => expect(canvas.dataset.worldStatus).toBe('ready'))

    fireEvent(canvas, new Event('webglcontextlost', { cancelable: true }))
    expect(createWorld).toHaveBeenCalledOnce()

    cleanup()
    fireEvent(canvas, new Event('webglcontextrestored'))
    await act(async () => Promise.resolve())

    expect(createWorld).toHaveBeenCalledOnce()
    expect(stopReconciler).toHaveBeenCalledOnce()
  })

  it('applies live reduced-motion preference changes while retaining selection and leader conversation', async () => {
    publishAccessibleEntities()
    let changeListener: ((event: MediaQueryListEvent) => void) | undefined

    const media = {
      addEventListener: vi.fn((_type: string, listener: (event: MediaQueryListEvent) => void) => {
        changeListener = listener
      }),
      matches: false,
      media: '(prefers-reduced-motion: reduce)',
      onchange: null,
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      dispatchEvent: vi.fn()
    }

    vi.stubGlobal(
      'matchMedia',
      vi.fn(() => media)
    )
    render(<LunarCity onOpenMemoryGraph={vi.fn()} />)

    await waitFor(() => expect(screen.getByLabelText('Interactive 3D Lunar City').dataset.worldStatus).toBe('ready'))
    expect(setReducedMotion).toHaveBeenCalledWith(false)
    act(() => changeListener?.({ matches: true } as MediaQueryListEvent))
    await waitFor(() => expect(setReducedMotion).toHaveBeenLastCalledWith(true))
    await waitFor(() => expect(screen.getByText(/Reduced motion: destinations snap into place/i)).toBeTruthy())
    fireEvent.click(screen.getByRole('button', { name: /Session Pip.*Working.*Research Lab/i }))
    expect(screen.getByRole('region', { name: 'Lunar City worker controls' })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Talk to fox-scientist leader on source-a' }))
    await waitFor(() => expect(screen.getByRole('dialog', { name: 'fox-scientist leader conversation' })).toBeTruthy())

    act(() => changeListener?.({ matches: false } as MediaQueryListEvent))
    await waitFor(() => expect(setReducedMotion).toHaveBeenLastCalledWith(false))
    await waitFor(() => expect(screen.queryByText(/Reduced motion: destinations snap into place/i)).toBeNull())
  })

  it('opens one exact profile-owned leader session without changing the city camera surface', async () => {
    const identity = { connectionId: 'source-a', kind: 'profile' as const, profile: 'owl' }

    const profile = {
      animation: 'rest',
      authority: 'authoritative' as const,
      destination: 'garden' as const,
      identity,
      key: 'profile-source-a-owl' as never,
      observedAt: 42
    }

    $lunarCitySnapshot.set({
      entities: new Map([[profile.key, profile]]),
      observedAt: 42,
      revision: 1,
      sources: []
    })
    render(<LunarCity onOpenMemoryGraph={vi.fn()} />)

    fireEvent.click(screen.getByRole('button', { name: 'Talk to owl leader on source-a' }))

    await waitFor(() => expect(resolveLeaderSession).toHaveBeenCalledWith({ connectionId: 'source-a', profile: 'owl' }))
    expect(screen.getByRole('dialog', { name: 'owl leader conversation' })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Rotate Left' }))
    expect(dispatchCamera).toHaveBeenCalledWith({ kind: 'orbit', deltaAlpha: -0.22, deltaBeta: 0 })

    fireEvent.change(screen.getByRole('textbox', { name: 'Message owl leader' }), {
      target: { value: 'Continue safely' }
    })
    fireEvent.click(screen.getByRole('button', { name: 'Send message' }))
    await waitFor(() =>
      expect(requestForSessionProfile).toHaveBeenCalledWith(
        { connectionId: 'source-a', profile: 'owl' },
        expect.any(Function),
        'prompt.submit',
        { session_id: 'runtime-owl', text: 'Continue safely' }
      )
    )
    expect(createWorld).toHaveBeenCalledOnce()
  })

  it('projects an exact session-resolution failure to only that leader unavailable clip', async () => {
    const identity = { connectionId: 'source-a', kind: 'profile' as const, profile: 'owl' }

    const profile = {
      animation: 'rest',
      authority: 'authoritative' as const,
      destination: 'garden' as const,
      identity,
      key: 'profile-source-a-owl' as never,
      observedAt: 42
    }

    resolveLeaderSession.mockRejectedValueOnce(new Error('owner route unavailable'))
    $lunarCitySnapshot.set({
      entities: new Map([[profile.key, profile]]),
      observedAt: 42,
      revision: 1,
      sources: []
    })
    render(<LunarCity onOpenMemoryGraph={vi.fn()} />)

    fireEvent.click(screen.getByRole('button', { name: 'Talk to owl leader on source-a' }))

    await waitFor(() => {
      expect(screen.getByRole('status', { name: 'owl leader conversation error' }).textContent).toContain(
        'owner route unavailable'
      )
      expect(leaderAnimation).toHaveBeenCalledWith(
        leaderModelIdForOwner({ connectionId: 'source-a', profile: 'owl' }),
        'unavailable'
      )
    })
  })

  it('opens the exact profile owner when its physical leader model is picked', async () => {
    publishAccessibleEntities()
    render(<LunarCity onOpenMemoryGraph={vi.fn()} />)
    await waitFor(() => expect(screen.getByLabelText('Interactive 3D Lunar City').dataset.worldStatus).toBe('ready'))

    act(() =>
      emitWorldIntent({
        entityKey:
          `lunar-city:leader:${leaderModelIdForOwner({ connectionId: 'source-a', profile: 'fox-scientist' })}` as never,
        kind: 'select-focus'
      })
    )

    await waitFor(() =>
      expect(resolveLeaderSession).toHaveBeenCalledWith({ connectionId: 'source-a', profile: 'fox-scientist' })
    )
    expect(dispatchCamera).toHaveBeenCalledWith({
      entityKey: `lunar-city:leader:${leaderModelIdForOwner({ connectionId: 'source-a', profile: 'fox-scientist' })}`,
      follow: false,
      kind: 'focus'
    })
    expect(screen.getByRole('dialog', { name: 'fox-scientist leader conversation' })).toBeTruthy()
  })

  it('requires exact-profile disambiguation when one physical leader model has multiple owners', async () => {
    const first = { connectionId: 'source-0', profile: 'duplicate' }
    const second = { connectionId: 'source-2', profile: 'duplicate' }
    const modelId = leaderModelIdForOwner(first)
    expect(leaderModelIdForOwner(second)).toBe(modelId)

    const entities = [first, second].map(owner => {
      const identity = { ...owner, kind: 'profile' as const }
      const key = entityKey(identity)

      return [
        key,
        {
          animation: 'rest' as const,
          authority: 'authoritative' as const,
          destination: 'council' as const,
          identity,
          key,
          observedAt: 42
        }
      ] as const
    })

    $lunarCitySnapshot.set({ entities: new Map(entities), observedAt: 42, revision: 1, sources: [] })
    render(<LunarCity onOpenMemoryGraph={vi.fn()} />)
    await waitFor(() => expect(screen.getByLabelText('Interactive 3D Lunar City').dataset.worldStatus).toBe('ready'))

    act(() => emitWorldIntent({ entityKey: `lunar-city:leader:${modelId}` as never, kind: 'select-focus' }))

    expect(resolveLeaderSession).not.toHaveBeenCalled()
    const chooser = screen.getByRole('region', { name: `Choose exact ${modelId} profile` })
    fireEvent.click(
      chooser.querySelector(`button[aria-label="Talk to ${second.profile} leader on ${second.connectionId}"]`)!
    )
    await waitFor(() => expect(resolveLeaderSession).toHaveBeenCalledWith(second))
  })

  it('routes the production accessible model selector through exact-owner disambiguation and camera focus', async () => {
    const first = { connectionId: 'source-0', profile: 'duplicate' }
    const second = { connectionId: 'source-2', profile: 'duplicate' }
    const modelId = leaderModelIdForOwner(first)
    expect(leaderModelIdForOwner(second)).toBe(modelId)

    const entities = [first, second].map(owner => {
      const identity = { ...owner, kind: 'profile' as const }
      const key = entityKey(identity)

      return [
        key,
        {
          animation: 'rest' as const,
          authority: 'authoritative' as const,
          destination: 'council' as const,
          identity,
          key,
          observedAt: 42
        }
      ] as const
    })

    $lunarCitySnapshot.set({ entities: new Map(entities), observedAt: 42, revision: 1, sources: [] })
    getCameraState.mockReturnValue({
      focusedEntityKey: `lunar-city:leader:${modelId}` as never,
      following: false
    })
    render(<LunarCity onOpenMemoryGraph={vi.fn()} />)
    await waitFor(() => expect(screen.getByLabelText('Interactive 3D Lunar City').dataset.worldStatus).toBe('ready'))

    fireEvent.click(screen.getByRole('button', { name: `Select ${modelId} leader model with 2 exact profiles` }))

    const chooser = screen.getByRole('region', { name: `Choose exact ${modelId} profile` })
    expect(within(chooser).getByRole('button', { name: 'Talk to duplicate leader on source-0' })).toBeTruthy()
    fireEvent.click(within(chooser).getByRole('button', { name: 'Talk to duplicate leader on source-2' }))

    await waitFor(() => expect(resolveLeaderSession).toHaveBeenCalledWith(second))
    expect(dispatchCamera).toHaveBeenCalledWith({
      entityKey: `lunar-city:leader:${modelId}`,
      follow: false,
      kind: 'focus'
    })
    expect(screen.getByRole('status', { name: 'Camera position' }).textContent).toContain('duplicate leader')
  })
})

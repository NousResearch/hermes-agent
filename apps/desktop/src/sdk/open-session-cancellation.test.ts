import type { ReadableAtom } from 'nanostores'
// Real imported SDK, hydration and native open-session branching. Gateway waits,
// transcript publication and tab primitives are controlled; this is not a GUI test.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { deferred } from '../test/deferred'

vi.mock('@/app/chat/session-view', async () => {
  const { atom } = await import('nanostores')

  return { PRIMARY_SESSION_VIEW: { $awaitingResponse: atom(false), $busy: atom(false) } }
})
vi.mock('@/components/pane-shell/tree/store', async () => {
  const { atom } = await import('nanostores')

  return {
    $activeTreeGroup: atom(null),
    $collapsedTreeSides: atom(new Set()),
    $hiddenTreePanes: atom(new Set()),
    $layoutTree: atom(null),
    $narrowViewport: atom(false)
  }
})
vi.mock('@/contrib/events', () => ({ onGatewayEvent: vi.fn() }))
vi.mock('@/hermes', () => ({ deleteProfile: vi.fn(), getLogs: vi.fn(), getStatus: vi.fn(), hermesApi: vi.fn() }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))
vi.mock('@/store/system-actions', () => ({ runGatewayRestart: vi.fn() }))
vi.mock('@/store/session', async () => {
  const { atom } = await import('nanostores')

  return {
    $activeSessionId: atom(null),
    $connection: atom(null),
    $cronSessions: atom([]),
    $currentCwd: atom(''),
    $currentModel: atom(''),
    $gatewayState: atom('open'),
    $messages: atom([]),
    $messagingSessions: atom([]),
    $selectedStoredSessionId: atom(null),
    $sessions: atom([]),
    $unreadFinishedSessionIds: atom([]),
    getSessionOwnerHints: () => [],
    lineageAliases: (id: string) => [id],
    rememberedSessionProfile: (_rows: unknown, _id: unknown, profile: string) => profile,
    markSessionRead: vi.fn(),
    requestSessionResume: vi.fn(),
    sessionMatchesStoredId: (row: { id: string }, id: string) => row.id === id,
    sessionPinId: (row: { id: string }) => row.id,
    setSessionOwnerHint: vi.fn(),
    setResumeExhaustedSessionId: vi.fn()
  }
})
vi.mock('@/store/session-focus', async importOriginal => {
  const { atom } = await import('nanostores')

  return { ...(await importOriginal<Record<string, unknown>>()), $focusedStoredSessionId: atom(null) }
})
vi.mock('@/store/session-states', async () => {
  const { atom } = await import('nanostores')

  return {
    $attentionSessionIds: atom([]),
    $draftSessionIds: atom([]),
    $focusedRuntimeId: atom(null),
    $focusedSessionState: atom(null),
    $sessionTiles: atom([]),
    $sessionStates: atom({}),
    $stalledSessionIds: atom([]),
    $workingSessionIds: atom([]),
    focusedSessionNeedsRoute: vi.fn(() => false),
    focusOpenSession: vi.fn(),
    frontMainIfSelected: vi.fn(),
    openSessionTile: vi.fn(),
    reuseBlankDraftTile: vi.fn(() => false),
    setSessionTileWorkspaceScope: vi.fn(),
    dropTilesForProfile: vi.fn(),
    sessionTileDelegate: vi.fn(() => null)
  }
})
vi.mock('@/store/profile', async () => {
  const { atom } = await import('nanostores')

  return {
    $activeGatewayProfile: atom('alpha'),
    $gatewaySwapTarget: atom(null),
    $hydrationSyncProfile: atom(null),
    $profiles: atom([]),
    $showAllProfiles: atom(false),
    ensureGatewayAgent: vi.fn(),
    ensureGatewayProfile: vi.fn(),
    newSessionInAgent: vi.fn(),
    newSessionInProfile: vi.fn(),
    normalizeProfileKey: (value: null | string | undefined) => value?.trim() || 'default',
    refreshProfiles: vi.fn(async () => []),
    selectProfile: vi.fn(),
    setActiveProfile: vi.fn(),
    setShowAllProfiles: vi.fn()
  }
})
vi.mock('@/store/gateway', async () => {
  const { atom } = await import('nanostores')

  return {
    $activeGatewayRoute: atom('default'),
    $gateway: atom(null),
    activeGateway: vi.fn(() => null),
    activeGatewayConnectionId: vi.fn(() => 'local'),
    ensureGatewayForAgent: vi.fn(),
    openGatewayForAgent: vi.fn(),
    openGatewayForProfile: vi.fn(),
    requestGatewayForAgent: vi.fn(),
    requestGatewayForProfile: vi.fn(),
    retainGatewayForAgent: vi.fn(async () => vi.fn()),
    retireLocalProfileGateways: vi.fn()
  }
})

const { host } = await import('./index')
const { $activeGatewayProfile, $gatewaySwapTarget, $hydrationSyncProfile, ensureGatewayProfile, setShowAllProfiles } =
  await import('@/store/profile')
const { openGatewayForAgent } = await import('@/store/gateway')
const {
  $selectedStoredSessionId,
  $activeSessionId,
  $messages,
  setSessionOwnerHint,
  requestSessionResume,
  setResumeExhaustedSessionId,
  markSessionRead
} = await import('@/store/session')
const { $focusedStoredSessionId } = await import('@/store/session-focus')
const {
  $sessionTiles,
  $focusedRuntimeId,
  $focusedSessionState,
  $sessionStates,
  sessionTileDelegate,
  focusOpenSession,
  openSessionTile
} = await import('@/store/session-states')
const { setWorkspaceScope, $workspaceOwnerKey } = await import('@/components/pane-shell/workspace-scope')

const route = { connectionId: 'source-a', profile: 'alpha', targetProfile: 'backend-alpha', mode: 'remote' as const }
const setAtom = <T>(store: unknown, value: T) => (store as { set(next: T): void }).set(value)
const flush = () => vi.advanceTimersByTimeAsync(0)
const outcome = (promise: Promise<void>) =>
  promise.then(
    () => 'opened',
    error => error.name
  )
const options = (signal?: AbortSignal) => ({
  route,
  signal,
  intent: 'tab' as const,
  awaitHydration: true,
  expectHistory: true,
  hydrationTimeoutMs: 100,
  retryHydrationTimeoutOnce: true
})

beforeEach(() => {
  vi.useFakeTimers()
  vi.mocked(openGatewayForAgent).mockResolvedValue(undefined)
  vi.mocked(sessionTileDelegate).mockReturnValue(null)
  vi.mocked(focusOpenSession).mockReturnValue(null)
  $activeGatewayProfile.set('alpha')
  setAtom($sessionTiles, [])
  setAtom($selectedStoredSessionId, null)
  setAtom($activeSessionId, null)
  setAtom($messages, [])
  setAtom($focusedStoredSessionId, null)
  setAtom($focusedRuntimeId, null)
  setAtom($focusedSessionState, null)
  setAtom($sessionStates, {})
  setWorkspaceScope('sessions')
})

afterEach(() => {
  vi.clearAllTimers()
  vi.useRealTimers()
  vi.restoreAllMocks()
  vi.clearAllMocks()
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

function paint(id: string, history = true) {
  setAtom($selectedStoredSessionId, id)
  setAtom($activeSessionId, `runtime-${id}`)
  setAtom($messages, history ? [{ id: 'history', parts: [], role: 'assistant' }] : [])
}

describe('caller-revocable SDK session opens', () => {
  it.each([false, true])(
    'aborts a stalled activation promptly (hydration=%s), observes late rejection and releases its timer/listener',
    async awaitHydration => {
      const dial = deferred<void>()
      vi.mocked(openGatewayForAgent).mockReturnValueOnce(dial.promise)
      const controller = new AbortController()
      const add = vi.spyOn(controller.signal, 'addEventListener')
      const remove = vi.spyOn(controller.signal, 'removeEventListener')
      const opening = outcome(host.openSession('old', { ...options(controller.signal), awaitHydration }))
      await flush()
      expect(openGatewayForAgent).toHaveBeenCalledWith('source-a', 'alpha', { spawnPriority: 'foreground' })
      controller.abort()
      await flush()
      expect(await Promise.race([opening, Promise.resolve('pending')])).toBe('AbortError')
      expect(openSessionTile).not.toHaveBeenCalled()
      expect(markSessionRead).not.toHaveBeenCalled()
      expect(vi.getTimerCount()).toBe(0)
      expect(remove.mock.calls.length).toBe(add.mock.calls.length)
      dial.reject(new Error('abandoned dial failed'))
      await flush()
      expect(setResumeExhaustedSessionId).not.toHaveBeenCalled()
    }
  )

  it('pre-abort is mutation-free and cannot supersede a pending valid open', async () => {
    const dial = deferred<void>()
    vi.mocked(openGatewayForAgent).mockReturnValueOnce(dial.promise)
    const valid = host.openSession('winner', { route, intent: 'tab' })
    await flush()
    const rejected = new AbortController()
    rejected.abort()
    const calls = vi.mocked(setSessionOwnerHint).mock.calls.length
    setAtom($hydrationSyncProfile, 'sentinel')
    const invalid = outcome(
      host.openSession('invalid', { ...options(rejected.signal), workspaceMode: 'bots', workspaceOwnerKey: 'invalid' })
    )
    await flush()
    expect(await invalid).toBe('AbortError')
    expect($hydrationSyncProfile.get()).toBe('sentinel')
    expect($workspaceOwnerKey.get()).not.toBe('invalid')
    expect(vi.mocked(setSessionOwnerHint).mock.calls).toHaveLength(calls)
    dial.resolve()
    await valid
    expect(openSessionTile).toHaveBeenCalledExactlyOnceWith('winner', 'center')
  })

  it('rechecks synchronously before the native commit when a view subscriber aborts', async () => {
    const controller = new AbortController()
    vi.mocked(setShowAllProfiles).mockImplementationOnce(() => controller.abort())
    const opening = outcome(host.openSession('old', options(controller.signal)))
    await flush()
    expect(await opening).toBe('AbortError')
    expect(openSessionTile).not.toHaveBeenCalled()
    expect(markSessionRead).not.toHaveBeenCalled()
    expect(vi.getTimerCount()).toBe(0)
  })

  it('cancels hydration without rolling back a committed tab and removes hydration subscriptions', async () => {
    const controller = new AbortController()
    const unbinds: Array<ReturnType<typeof vi.fn>> = []

    for (const store of [
      $activeGatewayProfile,
      $selectedStoredSessionId,
      $activeSessionId,
      $messages,
      $focusedStoredSessionId,
      $focusedRuntimeId,
      $focusedSessionState,
      $sessionTiles,
      $sessionStates
    ]) {
      const readable = store as ReadableAtom<unknown>
      const listen = readable.listen.bind(readable)
      vi.spyOn(readable, 'listen').mockImplementation(callback => {
        const unbind = vi.fn(listen(callback))
        unbinds.push(unbind)

        return unbind
      })
    }

    const opening = outcome(host.openSession('committed', options(controller.signal)))
    await flush()
    expect(openSessionTile).toHaveBeenCalledExactlyOnceWith('committed', 'center')
    expect($gatewaySwapTarget.get()).toBe('alpha')
    controller.abort()
    await flush()
    expect(await opening).toBe('AbortError')
    expect(unbinds.length).toBeGreaterThan(0)
    expect(unbinds.every(unbind => unbind.mock.calls.length === 1)).toBe(true)
    expect($gatewaySwapTarget.get()).toBeNull()
    expect(vi.getTimerCount()).toBe(0)
    paint('committed')
    await vi.advanceTimersByTimeAsync(1000)
    expect(openSessionTile).toHaveBeenCalledTimes(1)
    expect(setResumeExhaustedSessionId).not.toHaveBeenCalled()
  })

  it('aborts a stalled tile refresh without falling back to a late resume or timeout', async () => {
    const refresh = deferred<string>()
    vi.mocked(sessionTileDelegate).mockReturnValue({ resumeTile: vi.fn(() => refresh.promise) } as never)
    setAtom($sessionTiles, [{ storedSessionId: 'old' }])
    const controller = new AbortController()
    const opening = outcome(host.openSession('old', options(controller.signal)))
    await flush()
    const resumes = vi.mocked(requestSessionResume).mock.calls.length
    controller.abort()
    await flush()
    expect(await Promise.race([opening, Promise.resolve('pending')])).toBe('AbortError')
    refresh.reject(new Error('late refresh rejection'))
    await flush()
    expect(requestSessionResume).toHaveBeenCalledTimes(resumes)
    expect(setResumeExhaustedSessionId).not.toHaveBeenCalled()
    expect(vi.getTimerCount()).toBe(0)
  })

  it.each(['abort', 'newer-open'])('revokes retry authority across a held route-registry check (%s)', async action => {
    const registry = deferred<unknown[]>()

    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
      getProfileRoutes: vi.fn(() => registry.promise)
    }
    const controller = new AbortController()
    const opening = outcome(host.openSession('old', options(controller.signal)))
    await flush()
    await vi.advanceTimersByTimeAsync(100)
    expect(window.hermesDesktop?.getProfileRoutes).toHaveBeenCalledOnce()

    if (action === 'abort') {
      controller.abort()
    } else {
      await host.openSession('new', { intent: 'tab' })
    }

    await flush()

    if (action === 'abort') {
      expect(await Promise.race([opening, Promise.resolve('pending')])).toBe('AbortError')
    }
    registry.resolve([route])
    await flush()
    expect(await opening).not.toBe('opened')
    expect(vi.mocked(openSessionTile).mock.calls.filter(([id]) => id === 'old')).toHaveLength(1)
    expect(setResumeExhaustedSessionId).not.toHaveBeenCalled()
    expect(vi.getTimerCount()).toBe(0)
  })

  it.each([false, true])(
    'retains cleanup authority for hydration started by an overlay teardown subscriber (signal=%s)',
    async withSignal => {
      const controller = new AbortController()
      const first = outcome(host.openSession('old', { ...options(controller.signal), route: undefined }))
      await flush()
      expect($gatewaySwapTarget.get()).toBe('alpha')

      let second: Promise<void> | undefined

      const unbind = $gatewaySwapTarget.listen(target => {
        if (target === null && !second) {
          second = host.openSession('new', {
            ...options(withSignal ? new AbortController().signal : undefined),
            route: undefined
          })
        }
      })

      try {
        controller.abort()
        await flush()
        expect(await first).toBe('AbortError')
        expect(second).toBeDefined()
        expect($gatewaySwapTarget.get()).toBe('alpha')
        expect(vi.mocked(openSessionTile).mock.calls.filter(([id]) => id === 'new')).toHaveLength(1)
      } finally {
        unbind()
      }

      paint('new')
      await second
      expect($gatewaySwapTarget.get()).toBeNull()
      expect(vi.getTimerCount()).toBe(0)
    }
  )

  it('cleans the overlay for a sequential signal-less open after cancellation', async () => {
    const controller = new AbortController()
    const first = outcome(host.openSession('old', { ...options(controller.signal), route: undefined }))
    await flush()
    controller.abort()
    await flush()
    expect(await first).toBe('AbortError')
    const second = host.openSession('new', { ...options(), route: undefined })
    paint('new')
    await second
    expect($gatewaySwapTarget.get()).toBeNull()
    expect(vi.getTimerCount()).toBe(0)
  })

  it('older cancellation does not clear a newer same-profile overlay or workspace scope', async () => {
    const a = new AbortController(),
      b = new AbortController()
    const first = outcome(
      host.openSession('a', { ...options(a.signal), workspaceMode: 'bots', workspaceOwnerKey: 'owner-a' })
    )
    await flush()
    const second = outcome(
      host.openSession('b', { ...options(b.signal), workspaceMode: 'bots', workspaceOwnerKey: 'owner-b' })
    )
    await flush()
    a.abort()
    await flush()
    // Publishing B's workspace can supersede A before its signal fires.
    expect(await first).not.toBe('opened')
    expect($gatewaySwapTarget.get()).toBe('alpha')
    expect($workspaceOwnerKey.get()).toBe('owner-b')
    paint('b')
    await flush()
    expect(await second).toBe('opened')
    expect($gatewaySwapTarget.get()).toBeNull()
  })

  it('cleans a paint-first syncing badge on abort but cannot erase a newer same-profile badge', async () => {
    vi.mocked(ensureGatewayProfile).mockResolvedValue(undefined)
    $activeGatewayProfile.set('default')
    const a = new AbortController(),
      b = new AbortController()
    paint('a')
    await host.openSession('a', {
      ...options(a.signal),
      route: undefined,
      profile: 'alpha',
      keepAllProfilesScope: false
    })
    expect($hydrationSyncProfile.get()).toBe('alpha')
    paint('b')
    await host.openSession('b', {
      ...options(b.signal),
      route: undefined,
      profile: 'alpha',
      keepAllProfilesScope: false
    })
    a.abort()
    expect($hydrationSyncProfile.get()).toBe('alpha')
    b.abort()
    expect($hydrationSyncProfile.get()).toBeNull()
    expect(vi.getTimerCount()).toBe(0)
  })

  it('preserves signal-less native tab and exact route behavior, including an explicit workspace switch', async () => {
    vi.mocked(focusOpenSession).mockReturnValueOnce('tile')
    paint('existing')
    await host.openSession('existing', options())
    expect(openSessionTile).not.toHaveBeenCalled()
    expect(setSessionOwnerHint).toHaveBeenCalledWith('existing', route)
    await host.openSession('empty', { route, intent: 'tab', expectHistory: false })
    expect(openSessionTile).toHaveBeenCalledExactlyOnceWith('empty', 'center')
    vi.mocked(ensureGatewayProfile).mockImplementationOnce(async profile => {
      $activeGatewayProfile.set(profile || 'default')
    })
    await host.openSession('workspace', { profile: 'beta', keepAllProfilesScope: false })
    expect($activeGatewayProfile.get()).toBe('beta')
    expect(setShowAllProfiles).toHaveBeenLastCalledWith(false)
  })
})

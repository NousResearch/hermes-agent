// Stale model-pick ordering: a superseded selectModel (newer pick, or a chat
// switch while config.set is in flight) must never roll the pill or a tile back.
import { QueryClient } from '@tanstack/react-query'
import { cleanup, render, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { modelOptionsQueryKey } from '@/lib/model-options'
import { confirm } from '@/store/confirm'
import { $activeGatewayProfile } from '@/store/profile'
import {
  $activeSessionId,
  $currentModel,
  $currentProvider,
  markComposerSelectionManual,
  setCurrentModel,
  setCurrentModelSource,
  setCurrentProvider,
  setCurrentReasoningEffort
} from '@/store/session'
import * as SessionStates from '@/store/session-states'

import { deferred } from '../../../test/deferred'

import { useModelControls } from './use-model-controls'

const tile = vi.hoisted(() => ({ delegate: null as unknown }))
const gateway = vi.hoisted(() => ({ activeConnectionId: null as null | string }))
const notifyError = vi.fn()

vi.mock('@/hermes', () => ({
  getGlobalModelInfo: vi.fn(),
  setApiRequestProfile: vi.fn(),
  setGlobalModel: vi.fn()
}))

vi.mock('@/store/session-states', async importOriginal => {
  const actual = await importOriginal<typeof SessionStates>()

  return {
    ...actual,
    sessionTileDelegate: () => tile.delegate
  }
})

vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  activeGatewayConnectionId: () => gateway.activeConnectionId
}))

vi.mock('@/i18n', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useI18n: () => ({
    t: {
      desktop: {
        modelSwitchFailed: 'Model switch failed'
      }
    }
  })
}))

vi.mock('@/store/confirm', () => ({
  confirm: vi.fn()
}))

vi.mock('@/store/notifications', () => ({
  dismissNotification: vi.fn(),
  notify: vi.fn(),
  notifyError: (...args: Parameters<typeof notifyError>) => notifyError(...args)
}))

type Controls = ReturnType<typeof useModelControls>

function Harness({
  onReady,
  requestGateway
}: {
  onReady: (controls: Controls) => void
  requestGateway: <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>
}) {
  const controls = useModelControls({
    queryClient: new QueryClient(),
    requestGateway
  })

  onReady(controls)

  return null
}

function resetStores() {
  $activeGatewayProfile.set('default')
  $activeSessionId.set(null)
  setCurrentModel('')
  setCurrentModelSource('')
  setCurrentProvider('')
  SessionStates.$sessionStates.set({})
  SessionStates.$sessionTiles.set([])
  gateway.activeConnectionId = null
}

describe('useModelControls stale pick ordering', () => {
  beforeEach(() => {
    notifyError.mockReset()
    resetStores()
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    resetStores()
  })

  it('does not roll a failed older pick back over a newer one that applied', async () => {
    $activeSessionId.set('session-1')
    setCurrentModel('fable-5')
    setCurrentProvider('nous')

    const older = deferred<never>()
    const newer = deferred<unknown>()
    const requestGateway = vi.fn().mockReturnValueOnce(older.promise).mockReturnValueOnce(newer.promise)
    let controls!: Controls

    render(<Harness onReady={value => (controls = value)} requestGateway={requestGateway} />)

    const olderPick = controls.selectModel({ model: 'grok-4.5', provider: 'xai' })
    const newerPick = controls.selectModel({ model: 'gpt-6.1-luna', provider: 'openai' })

    newer.resolve({})
    await expect(newerPick).resolves.toBe(true)
    older.reject(new Error('timeout'))
    await expect(olderPick).resolves.toBe(false)

    expect($currentModel.get()).toBe('gpt-6.1-luna')
    expect($currentProvider.get()).toBe('openai')
    // The abandoned pick's failure is not the user's current choice: no error toast.
    expect(notifyError).not.toHaveBeenCalled()
  })

  it('does not paint the previous session rollback into a session opened mid-switch', async () => {
    $activeSessionId.set('session-1')
    setCurrentModel('fable-5')
    setCurrentProvider('nous')

    const pending = deferred<never>()
    const requestGateway = vi.fn(() => pending.promise)
    let controls!: Controls

    render(<Harness onReady={value => (controls = value)} requestGateway={requestGateway} />)

    const pick = controls.selectModel({ model: 'grok-4.5', provider: 'xai' })

    // The user opens another chat; its session.info paints that chat's model.
    $activeSessionId.set('session-2')
    setCurrentModel('gpt-6.1-luna')
    setCurrentProvider('openai')

    pending.reject(new Error('timeout'))
    await expect(pick).resolves.toBe(false)

    expect($currentModel.get()).toBe('gpt-6.1-luna')
    expect($currentProvider.get()).toBe('openai')
    expect(notifyError).not.toHaveBeenCalled()
  })

  it('does not roll a reused runtime id back after its profile owner changes', async () => {
    $activeGatewayProfile.set('alpha')
    $activeSessionId.set('shared-runtime')
    setCurrentModel('alpha-old')
    setCurrentProvider('alpha-provider')

    const pending = deferred<never>()
    const requestGateway = vi.fn(() => pending.promise)
    let controls!: Controls

    render(<Harness onReady={value => (controls = value)} requestGateway={requestGateway} />)

    const pick = controls.selectModel({ model: 'alpha-new', provider: 'alpha-provider' })

    $activeGatewayProfile.set('beta')
    setCurrentModel('beta-model')
    setCurrentProvider('beta-provider')

    pending.reject(new Error('timeout'))
    await expect(pick).resolves.toBe(false)

    expect($currentModel.get()).toBe('beta-model')
    expect($currentProvider.get()).toBe('beta-provider')
    expect(notifyError).not.toHaveBeenCalled()
  })

  it('does not roll back a confirmed request that fails after its profile owner changes', async () => {
    $activeGatewayProfile.set('alpha')
    $activeSessionId.set('shared-runtime')
    setCurrentModel('alpha-old')
    setCurrentProvider('alpha-provider')
    vi.mocked(confirm).mockResolvedValueOnce(true)

    const confirmed = deferred<never>()
    const later = deferred<never>()

    const requestGateway = vi
      .fn()
      .mockResolvedValueOnce({ confirm_required: true, confirm_message: 'Expensive model' })
      .mockReturnValueOnce(confirmed.promise)
      .mockReturnValueOnce(later.promise)

    let controls!: Controls

    render(<Harness onReady={value => (controls = value)} requestGateway={requestGateway} />)

    await expect(controls.selectModel({ model: 'alpha-new', provider: 'alpha-provider' })).resolves.toBe(false)
    await vi.waitFor(() => expect(requestGateway).toHaveBeenCalledTimes(2))

    $activeGatewayProfile.set('beta')
    setCurrentModel('beta-model')
    setCurrentProvider('beta-provider')

    confirmed.reject(new Error('timeout'))

    await confirmed.promise.catch(() => undefined)
    await new Promise<void>(resolve => queueMicrotask(resolve))

    expect($currentModel.get()).toBe('beta-model')
    expect($currentProvider.get()).toBe('beta-provider')
    expect(notifyError).not.toHaveBeenCalled()

    $activeGatewayProfile.set('alpha')
    setCurrentModel('alpha-old')
    setCurrentProvider('alpha-provider')

    const laterPick = controls.selectModel({ model: 'alpha-later', provider: 'alpha-provider' })

    later.reject(new Error('later timeout'))
    await expect(laterPick).resolves.toBe(false)

    expect($currentModel.get()).toBe('alpha-old')
    expect($currentProvider.get()).toBe('alpha-provider')
  })

  it('keeps a same-runtime same-target pick session-scoped when the Desktop owner differs', async () => {
    $activeGatewayProfile.set('beta')
    $activeSessionId.set('shared-runtime')
    setCurrentModel('primary-model')
    setCurrentProvider('primary-provider')
    gateway.activeConnectionId = 'connection-b'
    SessionStates.$sessionStates.set({
      'shared-runtime': { model: 'beta-old', provider: 'beta-provider' }
    } as never)
    SessionStates.$sessionTiles.set([
      {
        ownerRoute: { connectionId: 'connection-b', profile: 'beta-desktop', targetProfile: 'beta' },
        runtimeId: 'shared-runtime',
        storedSessionId: 'stored-b'
      }
    ])

    const updateSession = vi.fn()
    const requestGateway = vi.fn().mockRejectedValue(new Error('timeout'))
    tile.delegate = { updateSession }

    const { result } = renderHook(() =>
      useModelControls({
        cacheOwnerConnectionId: 'connection-b',
        cacheOwnerProfile: 'beta-desktop',
        cacheProfile: 'beta',
        queryClient: new QueryClient(),
        requestGateway
      })
    )

    try {
      await expect(
        result.current.selectModel({ model: 'beta-new', provider: 'beta-provider', sessionId: 'shared-runtime' })
      ).resolves.toBe(false)
    } finally {
      tile.delegate = null
    }

    expect($currentModel.get()).toBe('primary-model')
    expect($currentProvider.get()).toBe('primary-provider')
    expect(updateSession).toHaveBeenCalledTimes(2)
    expect(updateSession.mock.lastCall?.[1]({ model: 'beta-new', provider: 'beta-provider' })).toEqual(
      expect.objectContaining({ model: 'beta-old', provider: 'beta-provider' })
    )
    expect(notifyError).toHaveBeenCalledOnce()
    expect(requestGateway).toHaveBeenCalledWith(
      'config.set',
      expect.objectContaining({ value: 'beta-new --provider beta-provider --session' })
    )
  })

  // Two overlapping picks A then B from X: B's own pre-pick snapshot is A (painted, never applied),
  // so the pill must settle on the last CONFIRMED selection, whichever request fails first.
  interface InFlightPick {
    promise: Promise<boolean>
    reject: (reason?: unknown) => void
    resolve: (value: unknown) => void
  }

  async function overlappingPicks(settle: (a: InFlightPick, b: InFlightPick) => Promise<void>) {
    $activeSessionId.set('session-1')
    setCurrentModel('fable-5')
    setCurrentProvider('nous')

    const a = deferred<unknown>()
    const b = deferred<unknown>()
    const requestGateway = vi.fn().mockReturnValueOnce(a.promise).mockReturnValueOnce(b.promise)
    let controls!: Controls

    render(<Harness onReady={value => (controls = value)} requestGateway={requestGateway} />)

    const pickA = controls.selectModel({ model: 'grok-4.5', provider: 'xai' })
    const pickB = controls.selectModel({ model: 'gpt-6.1-luna', provider: 'openai' })

    await settle({ ...a, promise: pickA }, { ...b, promise: pickB })
  }

  it('settles on the confirmed model when the newer pick fails before the older one', async () => {
    await overlappingPicks(async (a, b) => {
      b.reject(new Error('timeout'))
      await expect(b.promise).resolves.toBe(false)
      a.reject(new Error('timeout'))
      await expect(a.promise).resolves.toBe(false)
    })

    expect($currentModel.get()).toBe('fable-5')
    expect($currentProvider.get()).toBe('nous')
    // Only the newest pick's failure is reported; the superseded one stays quiet.
    expect(notifyError).toHaveBeenCalledTimes(1)
  })

  it('settles on the confirmed model when the older pick fails before the newer one', async () => {
    await overlappingPicks(async (a, b) => {
      a.reject(new Error('timeout'))
      await expect(a.promise).resolves.toBe(false)
      b.reject(new Error('timeout'))
      await expect(b.promise).resolves.toBe(false)
    })

    expect($currentModel.get()).toBe('fable-5')
    expect($currentProvider.get()).toBe('nous')
  })

  it('shows the older pick when the newer one fails and the older one then applies', async () => {
    await overlappingPicks(async (a, b) => {
      b.reject(new Error('timeout'))
      await expect(b.promise).resolves.toBe(false)
      a.resolve({})
      await expect(a.promise).resolves.toBe(true)
    })

    expect($currentModel.get()).toBe('grok-4.5')
    expect($currentProvider.get()).toBe('xai')
  })

  // A confirm-gated pick leaves the ledger while its dialog is open; once confirmed and applied it
  // must count as the confirmed selection, or an older pick failing later restores the start.
  it('keeps a confirmed expensive pick when an older pick fails afterwards', async () => {
    $activeSessionId.set('session-1')
    setCurrentModel('fable-5')
    setCurrentProvider('nous')
    vi.mocked(confirm).mockResolvedValueOnce(true)

    const older = deferred<never>()
    const confirmed = deferred<unknown>()

    const requestGateway = vi
      .fn()
      .mockReturnValueOnce(older.promise)
      .mockResolvedValueOnce({ confirm_required: true, confirm_message: 'Expensive model' })
      .mockReturnValueOnce(confirmed.promise)

    let controls!: Controls

    render(<Harness onReady={value => (controls = value)} requestGateway={requestGateway} />)

    const olderPick = controls.selectModel({ model: 'grok-4.5', provider: 'xai' })
    await expect(controls.selectModel({ model: 'opus-9', provider: 'anthropic' })).resolves.toBe(false)
    await vi.waitFor(() => expect(requestGateway).toHaveBeenCalledTimes(3))
    confirmed.resolve({})
    await vi.waitFor(() => expect($currentModel.get()).toBe('opus-9'))

    older.reject(new Error('timeout'))
    await expect(olderPick).resolves.toBe(false)

    expect($currentModel.get()).toBe('opus-9')
    expect($currentProvider.get()).toBe('anthropic')
  })

  it('reconciles a confirmed pick that applies after a newer pick', async () => {
    $activeSessionId.set('session-1')
    setCurrentModel('fable-5')
    setCurrentProvider('nous')
    vi.mocked(confirm).mockResolvedValueOnce(true)

    const confirmed = deferred<unknown>()

    const requestGateway = vi
      .fn()
      .mockResolvedValueOnce({ confirm_required: true, confirm_message: 'Expensive model' })
      .mockReturnValueOnce(confirmed.promise)
      .mockResolvedValueOnce({})

    const queryClient = new QueryClient()
    const invalidateQueries = vi.spyOn(queryClient, 'invalidateQueries').mockResolvedValue()
    const { result } = renderHook(() => useModelControls({ queryClient, requestGateway }))

    await expect(result.current.selectModel({ model: 'opus-9', provider: 'anthropic' })).resolves.toBe(false)
    await vi.waitFor(() => expect(requestGateway).toHaveBeenCalledTimes(2))
    await expect(result.current.selectModel({ model: 'gpt-6.1-luna', provider: 'openai' })).resolves.toBe(true)
    invalidateQueries.mockClear()

    confirmed.resolve({})
    await vi.waitFor(() => expect(invalidateQueries).toHaveBeenCalledOnce())

    expect($currentModel.get()).toBe('gpt-6.1-luna')
    expect($currentProvider.get()).toBe('openai')
  })

  it('rolls a failed pick back even when the reasoning effort stepped mid-switch', async () => {
    $activeSessionId.set('session-1')
    setCurrentModel('fable-5')
    setCurrentProvider('nous')

    const pending = deferred<never>()
    const requestGateway = vi.fn(() => pending.promise)
    let controls!: Controls

    render(<Harness onReady={value => (controls = value)} requestGateway={requestGateway} />)

    const pick = controls.selectModel({ model: 'grok-4.5', provider: 'xai' })

    // The effort keybind (and the preset editor) bump the composer selection generation without
    // touching the model; that must not cost the failed pick its rollback.
    markComposerSelectionManual()
    setCurrentReasoningEffort('high')

    pending.reject(new Error('timeout'))
    await expect(pick).resolves.toBe(false)

    expect($currentModel.get()).toBe('fable-5')
    expect($currentProvider.get()).toBe('nous')
    expect(notifyError).toHaveBeenCalledTimes(1)
  })

  it('does not roll a failed older tile pick back over a newer one that applied', async () => {
    $activeSessionId.set('runtime-a')
    SessionStates.$sessionStates.set({ 'runtime-b': { model: 'old-b', provider: 'provider-b' } } as never)
    const updateSession = vi.fn()
    tile.delegate = { updateSession }

    const older = deferred<never>()
    const newer = deferred<unknown>()
    const requestGateway = vi.fn().mockReturnValueOnce(older.promise).mockReturnValueOnce(newer.promise)
    const { result } = renderHook(() => useModelControls({ queryClient: new QueryClient(), requestGateway }))

    const olderPick = result.current.selectModel({ model: 'grok-4.5', provider: 'xai', sessionId: 'runtime-b' })
    const newerPick = result.current.selectModel({ model: 'gpt-6.1-luna', provider: 'openai', sessionId: 'runtime-b' })

    try {
      newer.resolve({})
      await expect(newerPick).resolves.toBe(true)
      updateSession.mockClear()
      older.reject(new Error('timeout'))
      await expect(olderPick).resolves.toBe(false)
    } finally {
      tile.delegate = null
    }

    // The older failure must not repaint the tile with its pre-pick model.
    expect(updateSession).not.toHaveBeenCalled()
  })

  it('does not roll a failed tile pick into a reused runtime owned by another gateway', async () => {
    $activeSessionId.set('runtime-a')
    SessionStates.$sessionStates.set({ 'runtime-b': { model: 'old-b', provider: 'provider-b' } } as never)
    SessionStates.$sessionTiles.set([
      {
        ownerRoute: { connectionId: 'connection-b', profile: 'beta-desktop', targetProfile: 'beta' },
        runtimeId: 'runtime-b',
        storedSessionId: 'stored-b'
      }
    ])

    const updateSession = vi.fn()
    const pending = deferred<never>()
    const requestGateway = vi.fn(() => pending.promise)
    tile.delegate = { updateSession }

    const { result } = renderHook(() =>
      useModelControls({
        cacheOwnerConnectionId: 'connection-b',
        cacheOwnerProfile: 'beta-desktop',
        cacheProfile: 'beta',
        queryClient: new QueryClient(),
        requestGateway
      })
    )

    const pick = result.current.selectModel({ model: 'new-b', provider: 'provider-b', sessionId: 'runtime-b' })

    SessionStates.$sessionTiles.set([
      {
        ownerRoute: { connectionId: 'connection-c', profile: 'gamma-desktop', targetProfile: 'gamma' },
        runtimeId: 'runtime-b',
        storedSessionId: 'stored-c'
      }
    ])
    SessionStates.$sessionStates.set({ 'runtime-b': { model: 'gamma-model', provider: 'gamma-provider' } } as never)
    updateSession.mockClear()

    try {
      pending.reject(new Error('timeout'))
      await expect(pick).resolves.toBe(false)
    } finally {
      tile.delegate = null
    }

    expect(updateSession).not.toHaveBeenCalled()
    expect(notifyError).not.toHaveBeenCalled()
  })

  it('reconciles a confirmed tile pick only through its captured owner after runtime reuse', async () => {
    $activeSessionId.set('runtime-a')
    SessionStates.$sessionStates.set({ 'runtime-b': { model: 'old-b', provider: 'provider-b' } } as never)
    SessionStates.$sessionTiles.set([
      {
        ownerRoute: { connectionId: 'connection-b', profile: 'beta-desktop', targetProfile: 'beta' },
        runtimeId: 'runtime-b',
        storedSessionId: 'stored-b'
      }
    ])
    vi.mocked(confirm).mockResolvedValueOnce(true)

    const updateSession = vi.fn()
    const confirmed = deferred<unknown>()

    const requestGateway = vi
      .fn()
      .mockResolvedValueOnce({ confirm_required: true, confirm_message: 'Expensive model' })
      .mockReturnValueOnce(confirmed.promise)

    const queryClient = new QueryClient()
    const invalidateQueries = vi.spyOn(queryClient, 'invalidateQueries').mockResolvedValue()
    tile.delegate = { updateSession }

    const { result } = renderHook(() =>
      useModelControls({
        cacheOwnerConnectionId: 'connection-b',
        cacheOwnerProfile: 'beta-desktop',
        cacheProfile: 'beta',
        queryClient,
        requestGateway
      })
    )

    await expect(
      result.current.selectModel({ model: 'opus-9', provider: 'anthropic', sessionId: 'runtime-b' })
    ).resolves.toBe(false)
    await vi.waitFor(() => expect(requestGateway).toHaveBeenCalledTimes(2))

    SessionStates.$sessionTiles.set([
      {
        ownerRoute: { connectionId: 'connection-c', profile: 'gamma-desktop', targetProfile: 'gamma' },
        runtimeId: 'runtime-b',
        storedSessionId: 'stored-c'
      }
    ])
    SessionStates.$sessionStates.set({ 'runtime-b': { model: 'gamma-model', provider: 'gamma-provider' } } as never)
    updateSession.mockClear()
    invalidateQueries.mockClear()

    try {
      confirmed.resolve({})
      await vi.waitFor(() => expect(invalidateQueries).toHaveBeenCalledOnce())
    } finally {
      tile.delegate = null
    }

    expect(invalidateQueries).toHaveBeenCalledWith({
      queryKey: modelOptionsQueryKey('beta', 'runtime-b', 'connection-b')
    })
    expect(updateSession).not.toHaveBeenCalled()
    expect(notifyError).not.toHaveBeenCalled()
  })
})

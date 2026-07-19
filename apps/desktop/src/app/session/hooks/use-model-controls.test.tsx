import { QueryClient } from '@tanstack/react-query'
import { cleanup, render, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getGlobalModelInfo } from '@/hermes'
import {
  $activeSessionId,
  $currentModel,
  $currentProvider,
  getCurrentModelSource,
  setCurrentModel,
  setCurrentModelSource,
  setCurrentProvider
} from '@/store/session'

import { useModelControls } from './use-model-controls'

const setGlobalModel = vi.fn()
const notifyError = vi.fn()

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void

  const promise = new Promise<T>(done => {
    resolve = done
  })

  return { promise, resolve }
}

vi.mock('@/hermes', () => ({
  getGlobalModelInfo: vi.fn(),
  setGlobalModel: (...args: Parameters<typeof setGlobalModel>) => setGlobalModel(...args)
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      desktop: {
        modelSwitchFailed: 'Model switch failed'
      }
    }
  })
}))

vi.mock('@/store/notifications', () => ({
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

describe('useModelControls', () => {
  beforeEach(() => {
    $activeSessionId.set(null)
    setCurrentModel('')
    setCurrentModelSource('')
    setCurrentProvider('')
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    $activeSessionId.set(null)
    setCurrentModel('')
    setCurrentModelSource('')
    setCurrentProvider('')
  })

  it('applies the global model when there is no active runtime session', async () => {
    vi.mocked(getGlobalModelInfo).mockResolvedValue({
      model: 'openai/gpt-5.5',
      provider: 'openai-codex'
    })

    const { result } = renderHook(() =>
      useModelControls({
        queryClient: new QueryClient(),
        requestGateway: vi.fn()
      })
    )

    await result.current.refreshCurrentModel()

    expect($currentModel.get()).toBe('openai/gpt-5.5')
    expect($currentProvider.get()).toBe('openai-codex')
    expect(getCurrentModelSource()).toBe('default')
  })

  it('does not clobber the active session footer state with global model info', async () => {
    setCurrentModel('deepseek/deepseek-v4-pro')
    setCurrentProvider('deepseek')
    $activeSessionId.set('runtime-1')
    vi.mocked(getGlobalModelInfo).mockResolvedValue({
      model: 'openai/gpt-5.5',
      provider: 'openai-codex'
    })

    const { result } = renderHook(() =>
      useModelControls({
        queryClient: new QueryClient(),
        requestGateway: vi.fn()
      })
    )

    await result.current.refreshCurrentModel()

    expect($currentModel.get()).toBe('deepseek/deepseek-v4-pro')
    expect($currentProvider.get()).toBe('deepseek')
  })

  it('routes active-session picker changes through config.set with an explicit session-scoped provider', async () => {
    $activeSessionId.set('session-1')
    const requestGateway = vi.fn(async () => ({ key: 'model', value: 'claude-sonnet-4.6' }) as never)
    let controls!: Controls

    render(<Harness onReady={value => (controls = value)} requestGateway={requestGateway} />)

    await expect(
      controls.selectModel({
        model: 'claude-sonnet-4.6',
        provider: 'anthropic'
      })
    ).resolves.toBe(true)

    expect(requestGateway).toHaveBeenCalledWith('config.set', {
      session_id: 'session-1',
      key: 'model',
      value: 'claude-sonnet-4.6 --provider anthropic --session'
    })
    expect(requestGateway).not.toHaveBeenCalledWith('slash.exec', expect.anything())
  })

  it('session-scopes MoA preset selections so they cannot persist as the global gateway default', async () => {
    $activeSessionId.set('session-1')
    const requestGateway = vi.fn(async () => ({ key: 'model', value: 'BeastMode' }) as never)
    let controls!: Controls

    render(<Harness onReady={value => (controls = value)} requestGateway={requestGateway} />)

    await expect(
      controls.selectModel({
        model: 'BeastMode',
        provider: 'moa'
      })
    ).resolves.toBe(true)

    expect(requestGateway).toHaveBeenCalledWith('config.set', {
      session_id: 'session-1',
      key: 'model',
      value: 'BeastMode --provider moa --session'
    })
  })

  it('stores a no-session pick as UI state with no gateway or global write', async () => {
    const requestGateway = vi.fn()
    let controls!: Controls

    render(<Harness onReady={value => (controls = value)} requestGateway={requestGateway} />)

    await expect(
      controls.selectModel({
        model: 'claude-sonnet-4.6',
        provider: 'anthropic'
      })
    ).resolves.toBe(true)

    // The pick is plain UI state; session.create ships it later. Nothing touches
    // the gateway or the profile default here.
    expect($currentModel.get()).toBe('claude-sonnet-4.6')
    expect($currentProvider.get()).toBe('anthropic')
    expect(getCurrentModelSource()).toBe('manual')
    expect(requestGateway).not.toHaveBeenCalled()
    expect(setGlobalModel).not.toHaveBeenCalled()
  })

  it('seeds an empty composer model from global but never clobbers a pick', async () => {
    vi.mocked(getGlobalModelInfo).mockResolvedValue({ model: 'openai/gpt-5.5', provider: 'openai-codex' })

    const { result } = renderHook(() =>
      useModelControls({
        queryClient: new QueryClient(),
        requestGateway: vi.fn()
      })
    )

    // Empty → seeds the default.
    await result.current.refreshCurrentModel()
    expect($currentModel.get()).toBe('openai/gpt-5.5')

    // A user pick must survive the lifecycle refreshes that fire on boot / fresh
    // draft / session events.
    setCurrentModel('anthropic/claude-sonnet-4.6')
    setCurrentModelSource('manual')
    setCurrentProvider('anthropic')
    await result.current.refreshCurrentModel()
    expect($currentModel.get()).toBe('anthropic/claude-sonnet-4.6')

    // A profile swap forces a reseed to the new profile's default.
    await result.current.refreshCurrentModel(true)
    expect($currentModel.get()).toBe('openai/gpt-5.5')
  })

  it('reseeds a sticky manual pick that was removed from the catalog', async () => {
    vi.mocked(getGlobalModelInfo).mockResolvedValue({ model: 'openai/gpt-5.5', provider: 'openai-codex' })

    const queryClient = new QueryClient()
    queryClient.setQueryData(['model-options', 'global'], {
      providers: [{ models: ['openai/gpt-5.5'], name: 'OpenRouter', slug: 'openrouter' }]
    })

    // A manual pick whose model no longer exists on its provider.
    setCurrentModel('openrouter/owl-alpha')
    setCurrentProvider('openrouter')
    setCurrentModelSource('manual')

    const { result } = renderHook(() => useModelControls({ queryClient, requestGateway: vi.fn() }))

    await result.current.refreshCurrentModel()

    expect($currentModel.get()).toBe('openai/gpt-5.5')
    expect(getCurrentModelSource()).toBe('default')
  })

  it('keeps a sticky manual pick that is still in the catalog', async () => {
    vi.mocked(getGlobalModelInfo).mockResolvedValue({ model: 'openai/gpt-5.5', provider: 'openai-codex' })

    const queryClient = new QueryClient()
    queryClient.setQueryData(['model-options', 'global'], {
      providers: [{ models: ['openrouter/glm-4.7', 'openai/gpt-5.5'], name: 'OpenRouter', slug: 'openrouter' }]
    })

    setCurrentModel('openrouter/glm-4.7')
    setCurrentProvider('openrouter')
    setCurrentModelSource('manual')

    const { result } = renderHook(() => useModelControls({ queryClient, requestGateway: vi.fn() }))

    await result.current.refreshCurrentModel()

    expect($currentModel.get()).toBe('openrouter/glm-4.7')
    expect(getCurrentModelSource()).toBe('manual')
  })

  it('does not let a stale forced profile refresh overwrite a newer picker choice', async () => {
    const profileDefault = deferred<Awaited<ReturnType<typeof getGlobalModelInfo>>>()
    vi.mocked(getGlobalModelInfo).mockReturnValueOnce(profileDefault.promise)

    const { result } = renderHook(() =>
      useModelControls({
        queryClient: new QueryClient(),
        requestGateway: vi.fn()
      })
    )

    const pendingRefresh = result.current.refreshCurrentModel(true)
    expect(getGlobalModelInfo).toHaveBeenCalled()

    await expect(
      result.current.selectModel({
        model: 'claude-sonnet-4.6',
        provider: 'anthropic'
      })
    ).resolves.toBe(true)

    profileDefault.resolve({ model: 'gpt-5.5', provider: 'openai-codex' })
    await pendingRefresh

    expect($currentModel.get()).toBe('claude-sonnet-4.6')
    expect($currentProvider.get()).toBe('anthropic')
    expect(getCurrentModelSource()).toBe('manual')
  })

  it('does not let an older profile refresh overwrite a newer profile', async () => {
    const profileB = deferred<Awaited<ReturnType<typeof getGlobalModelInfo>>>()
    const profileC = deferred<Awaited<ReturnType<typeof getGlobalModelInfo>>>()
    vi.mocked(getGlobalModelInfo).mockReturnValueOnce(profileB.promise).mockReturnValueOnce(profileC.promise)

    const { result } = renderHook(() =>
      useModelControls({
        queryClient: new QueryClient(),
        requestGateway: vi.fn()
      })
    )

    const refreshB = result.current.refreshCurrentModel(true)
    const refreshC = result.current.refreshCurrentModel(true)

    profileC.resolve({ model: 'profile-c-model', provider: 'profile-c-provider' })
    await refreshC
    profileB.resolve({ model: 'profile-b-model', provider: 'profile-b-provider' })
    await refreshB

    expect($currentModel.get()).toBe('profile-c-model')
    expect($currentProvider.get()).toBe('profile-c-provider')
  })

  it('refreshes legacy/default-derived composer state from the profile default', async () => {
    setCurrentModel('openai/gpt-5.5')
    setCurrentProvider('nous')
    setCurrentModelSource('')
    vi.mocked(getGlobalModelInfo).mockResolvedValue({ model: 'gpt-5.5', provider: 'openai-codex' })

    const { result } = renderHook(() =>
      useModelControls({
        queryClient: new QueryClient(),
        requestGateway: vi.fn()
      })
    )

    expect(getCurrentModelSource()).toBe('')

    await result.current.refreshCurrentModel()

    expect(getGlobalModelInfo).toHaveBeenCalled()
    expect($currentModel.get()).toBe('gpt-5.5')
    expect($currentProvider.get()).toBe('openai-codex')
    expect(getCurrentModelSource()).toBe('default')
  })
})

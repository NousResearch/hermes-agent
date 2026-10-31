import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { $activeGatewayProfile } from '@/store/profile'
import {
  $currentModel,
  $currentProvider,
  clearComposerSelectionOwner,
  markComposerSelectionManual,
  setComposerSelectionOwner,
  setConnection,
  setCurrentModel,
  setCurrentModelSource,
  setCurrentModelTransient,
  setCurrentProvider,
  setCurrentProviderTransient,
  setSessions
} from '@/store/session'
import { $sessionTiles, clearAllSessionStates } from '@/store/session-states'

import { desktopSessionCreateParams } from './session-create-request'
import { applyRuntimeInfo, applyStoredSessionPreviewRuntimeInfo } from './utils'

import { useSessionActions } from './index'

vi.mock('@/store/profile', async original => ({
  ...(await original<Record<string, unknown>>()),
  ensureGatewayAgent: vi.fn(async () => undefined),
  ensureGatewayProfile: vi.fn(async () => undefined)
}))

beforeEach(() => {
  window.localStorage.clear()
  $activeGatewayProfile.set('default')
  setConnection({
    baseUrl: 'http://localhost:7070',
    isFullscreen: false,
    logs: [],
    mode: 'local',
    nativeOverlayWidth: 0,
    token: '',
    windowButtonPosition: null,
    wsUrl: 'ws://localhost:7070'
  })
  setCurrentModel('chosen-model')
  setCurrentProvider('custom:chosen-endpoint')
  markComposerSelectionManual()
})

afterEach(() => {
  cleanup()
  $sessionTiles.set([])
  clearAllSessionStates()
  setSessions([])
  $activeGatewayProfile.set('default')
  setCurrentModel('')
  setCurrentProvider('')
  setCurrentModelSource('')
  setConnection(null)
  window.localStorage.clear()
})

describe('new-chat model route after runtime metadata', () => {
  it('paints a resolved runtime without replacing the persisted user route used by the next chat', async () => {
    applyRuntimeInfo({ model: 'runtime-fallback', provider: 'custom' })

    expect($currentModel.get()).toBe('runtime-fallback')
    expect($currentProvider.get()).toBe('custom')
    expect(window.localStorage.getItem('hermes.desktop.composer.model')).toBe('chosen-model')
    expect(window.localStorage.getItem('hermes.desktop.composer.provider')).toBe('custom:chosen-endpoint')
    await expect(desktopSessionCreateParams('')).resolves.toMatchObject({
      model: 'chosen-model',
      provider: 'custom:chosen-endpoint'
    })

    const state = createClientSessionState()
    const ref = <T>(current: T) => ({ current })

    const requestGateway = vi.fn(async () => ({
      info: { model: 'tile-runtime-model', provider: 'custom' },
      session_id: 'created-runtime',
      stored_session_id: 'created-stored'
    }))

    const { result } = renderHook(() =>
      useSessionActions({
        activeSessionId: 'runtime-main',
        activeSessionIdRef: ref<string | null>('runtime-main'),
        busyRef: ref(false),
        creatingSessionRef: ref(false),
        ensureSessionState: () => state,
        getRouteToken: () => 'route',
        getRoutedStoredSessionId: () => null,
        navigate: vi.fn(),
        requestGateway: requestGateway as never,
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

    await act(() => result.current.openNewSessionTile('right', { route: null }))
    expect(requestGateway).toHaveBeenCalledWith(
      'session.create',
      expect.objectContaining({ model: 'chosen-model', provider: 'custom:chosen-endpoint' })
    )
    expect($currentModel.get()).toBe('runtime-fallback')

    act(() => result.current.startFreshSessionDraft())
    expect($currentModel.get()).toBe('chosen-model')
    expect($currentProvider.get()).toBe('custom:chosen-endpoint')
  })

  it('creates from the sticky route, not heartbeat, fallback or abandoned-resume display values', async () => {
    // Exercise named entries sharing a model, a native provider and a real
    // direct-custom choice. A runtime class cannot tell which one was picked.
    for (const provider of ['custom:first-endpoint', 'second-endpoint', 'anthropic', 'custom']) {
      setCurrentModel('chosen-model')
      setCurrentProvider(provider)
      markComposerSelectionManual()

      // Same transient paint used by useSessionStateCache on session.info.
      setCurrentModelTransient('fallback-model')
      setCurrentProviderTransient('custom')
      expect($currentModel.get()).toBe('fallback-model')
      await expect(desktopSessionCreateParams('')).resolves.toMatchObject({ model: 'chosen-model', provider })

      // A cold resume can paint a model with no provider and then be abandoned.
      applyStoredSessionPreviewRuntimeInfo({ model: 'preview-model' }, 'preview-stored')
      await expect(desktopSessionCreateParams('')).resolves.toMatchObject({ model: 'chosen-model', provider })
    }

    // Gateway activation publishes the exact owner before profile hydration.
    setComposerSelectionOwner('test-backend', 'other')
    $activeGatewayProfile.set('other')
    expect(await desktopSessionCreateParams('')).not.toHaveProperty('model')
    setCurrentModel('other-model')
    setCurrentProvider('custom:other-endpoint')
    markComposerSelectionManual()
    setCurrentProviderTransient('custom')
    await expect(desktopSessionCreateParams('')).resolves.toMatchObject({
      model: 'other-model',
      provider: 'custom:other-endpoint'
    })
    $activeGatewayProfile.set('default')
    setConnection(current => current)
    await expect(desktopSessionCreateParams('')).resolves.toMatchObject({ model: 'chosen-model', provider: 'custom' })

    clearComposerSelectionOwner()
    setCurrentModelTransient('unresolved-model')
    expect(await desktopSessionCreateParams('')).not.toHaveProperty('model')
    setConnection(current => current)

    // Settings-sourced values remain a display cache, not a launch override.
    setCurrentModelSource('default')
    const inherited = await desktopSessionCreateParams('')
    expect(inherited).not.toHaveProperty('model')
    expect(inherited).not.toHaveProperty('provider')
  })
})

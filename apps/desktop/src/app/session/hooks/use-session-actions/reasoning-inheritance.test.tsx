import { act, cleanup, render, renderHook, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ReasoningPill } from '@/app/chat/composer/reasoning-pill'
import { PRIMARY_SESSION_VIEW, SessionViewProvider } from '@/app/chat/session-view'
import { getHermesConfig } from '@/hermes'
import { createClientSessionState } from '@/lib/chat-runtime'
import { getComposerReasoningEffort, setComposerReasoningEffort } from '@/store/composer-reasoning-effort'
import { applyModelPreset } from '@/store/model-presets'
import { $activeGatewayProfile, $newChatProfile, $newChatRoute, ensureGatewayProfile } from '@/store/profile'
import {
  $activeSessionId,
  $selectedStoredSessionId,
  setComposerSelectionOwner,
  setCurrentModel,
  setCurrentModelSource,
  setCurrentProvider,
  setCurrentReasoningEffort,
  setCurrentReasoningEffortWire,
  setDefaultReasoningEffort,
  setSessions
} from '@/store/session'
import { clearAllSessionStates } from '@/store/session-states'

import { deferred } from '../../../../test/deferred'
import { useHermesConfig } from '../use-hermes-config'

import { applyRuntimeInfo } from './utils'

import { useSessionActions } from './index'

vi.mock('@/hermes', async original => ({
  ...(await original<Record<string, unknown>>()),
  getHermesConfig: vi.fn(),
  getHermesConfigDefaults: vi.fn(async () => ({}))
}))
vi.mock('@/store/profile', async original => ({
  ...(await original<Record<string, unknown>>()),
  ensureGatewayProfile: vi.fn(async () => undefined)
}))

function setup() {
  const activeSessionIdRef = { current: null as string | null }
  const states = new Map<string, ReturnType<typeof createClientSessionState>>()

  const request = vi.fn(
    async (method: string, _params?: Record<string, unknown>) =>
      (method === 'session.create' ? { session_id: 'runtime-new', stored_session_id: null, info: {} } : {}) as never
  )

  const { result } = renderHook(() => ({
    config: useHermesConfig({ activeSessionIdRef }),
    actions: useSessionActions({
      activeSessionId: null,
      activeSessionIdRef,
      busyRef: { current: false },
      creatingSessionRef: { current: false },
      ensureSessionState: () => createClientSessionState(null),
      getRouteToken: () => 'draft',
      getRoutedStoredSessionId: () => null,
      navigate: vi.fn(),
      requestGateway: request,
      resetViewSync: vi.fn(),
      routedSessionId: null,
      runtimeIdByStoredSessionIdRef: { current: new Map<string, string>() },
      selectedStoredSessionId: null,
      selectedStoredSessionIdRef: { current: null },
      sessionStateByRuntimeIdRef: { current: states },
      syncSessionStateToView: vi.fn(),
      updateSessionState: (id, updater) => {
        const next = updater(states.get(id) ?? createClientSessionState(null))
        states.set(id, next)

        return next
      }
    })
  }))

  return { result, request, activeSessionIdRef }
}

function showEffort() {
  render(
    <SessionViewProvider value={PRIMARY_SESSION_VIEW}>
      <ReasoningPill
        disabled={false}
        model={{
          canSwitch: true,
          model: 'test-model',
          provider: 'test-provider',
          reasoningMenuContent: <div>menu</div>
        }}
      />
    </SessionViewProvider>
  )

  return () => screen.getByTestId('reasoning-pill').textContent
}

beforeEach(() => {
  window.localStorage.clear()
  clearAllSessionStates()
  setSessions([])
  $activeSessionId.set(null)
  $selectedStoredSessionId.set(null)
  $newChatProfile.set(null)
  $newChatRoute.set(null)
  $activeGatewayProfile.set('profile-a')
  setComposerSelectionOwner('test-connection', 'profile-a')
  setComposerReasoningEffort('')
  setCurrentModelSource('')
  setCurrentReasoningEffort('')
  setDefaultReasoningEffort('')
  vi.clearAllMocks()
})
afterEach(cleanup)

describe('fresh-session reasoning inheritance', () => {
  it.each([
    ['primary', 'default'],
    ['split', 'default'],
    ['primary', 'manual'],
    ['split', 'manual']
  ] as const)('lets %s creation inherit after a profile switch with %s model source', async (path, source) => {
    const { result, request, activeSessionIdRef } = setup()
    activeSessionIdRef.current = 'runtime-old'
    applyRuntimeInfo({ reasoning_effort: 'low' })
    setComposerSelectionOwner('test-connection', 'profile-b')
    $activeGatewayProfile.set('profile-b')
    vi.mocked(getHermesConfig).mockResolvedValue({ agent: { reasoning_effort: 'high' } })
    await act(async () => {
      await result.current.config.refreshHermesConfig(true)
    })
    // Config does not reseed the visible cache while the old runtime is active.
    setCurrentModel('test-model')
    setCurrentProvider('test-provider')
    setCurrentModelSource(source)
    setCurrentReasoningEffortWire('low')
    act(() => result.current.actions.startFreshSessionDraft())
    const label = showEffort()
    expect.soft(label()).toBe('High')
    await act(async () => {
      if (path === 'primary') {
        await result.current.actions.createBackendSessionForSend()
      } else {
        await result.current.actions.openNewSessionTile('center', { listed: false, route: null })
      }
    })
    const params = request.mock.calls.find(([method]) => method === 'session.create')?.[1]
    expect(params).toMatchObject({ profile: 'profile-b', fast: false })
    expect(params).not.toHaveProperty('reasoning_effort')

    if (source === 'manual') {
      expect(params).toMatchObject({ model: 'test-model', provider: 'test-provider' })
    } else {
      expect(params).not.toHaveProperty('model')
    }
  })

  it.each(['high', 'none'])('keeps a deliberate %s pick through hydration and profile round trips', async effort => {
    const { result, request } = setup()
    await applyModelPreset({ effort }, { failMessage: 'failed', request, sessionId: null })
    // A runtime report uses the same setter but is not a new user selection.
    applyRuntimeInfo({ reasoning_effort: 'low' })
    setDefaultReasoningEffort('low')
    setComposerSelectionOwner('test-connection', 'profile-b')
    setDefaultReasoningEffort('low')
    const label = showEffort()
    expect(label()).toBe('Low')
    act(() => setComposerSelectionOwner('test-connection', 'profile-a'))
    expect.soft(label()).toBe(effort === 'none' ? 'Off' : 'High')
    const ready = deferred<void>()
    vi.mocked(ensureGatewayProfile).mockReturnValueOnce(ready.promise)
    let create!: Promise<string | null>
    act(() => {
      create = result.current.actions.createBackendSessionForSend()
    })
    act(() => setCurrentReasoningEffort('low'))
    ready.resolve()
    await act(async () => {
      await create
    })
    expect(request.mock.calls.find(([method]) => method === 'session.create')?.[1]).toMatchObject({
      reasoning_effort: effort
    })
  })
})

describe('best-effort effort persistence', () => {
  it.each(['', 'high'])('sends the visible pick when storage retains %s after a failed write', async persisted => {
    const { result, request } = setup()
    setComposerReasoningEffort(persisted)

    const storageWrite = vi.spyOn(window.localStorage, 'setItem').mockImplementation(() => {
      throw new DOMException('fixture storage unavailable', 'QuotaExceededError')
    })

    try {
      act(() => setComposerReasoningEffort('none'))
      expect(storageWrite).toHaveBeenCalled()
      const label = showEffort()
      expect.soft(label()).toBe('Off')
      expect.soft(getComposerReasoningEffort()).toBe('none')
      act(() => setComposerSelectionOwner('test-connection', 'profile-a'))
      expect.soft(label()).toBe('Off')
      await act(async () => {
        await result.current.actions.createBackendSessionForSend()
      })
      expect(request.mock.calls.find(([method]) => method === 'session.create')?.[1]).toMatchObject({
        reasoning_effort: 'none'
      })
    } finally {
      storageWrite.mockRestore()
    }
  })
})

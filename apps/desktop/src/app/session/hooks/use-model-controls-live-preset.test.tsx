import type { ModelOptionsResult } from '@hermes/shared'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { CanonicalDesktopProtocol } from '@/api/canonical-protocol'
import { PRIMARY_SESSION_VIEW } from '@/app/chat/session-view'
import { useModelMenuController } from '@/app/shell/use-model-menu-controller'
import { $modelPresets, setModelPreset } from '@/store/model-presets'
import { $activeGatewayProfile } from '@/store/profile'
import {
  $activeSessionId,
  $defaultReasoningEffort,
  setAwaitingResponse,
  setBusy,
  setCurrentModel,
  setCurrentProvider,
  setCurrentReasoningEffort,
  setMessages
} from '@/store/session'
import { $sessionStates, type SessionTileDelegate, setSessionTileDelegate } from '@/store/session-states'

import { deferred } from '../../../test/deferred'

import { sessionInfoStatePatch } from './use-message-stream/utils'
import { useModelControls } from './use-model-controls'
import { useSessionStateCache } from './use-session-state-cache'

vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))

const SID = 'live-preset-session'
const A = { model: 'model-a', provider: 'provider-a' }
const B = { model: 'model-b', provider: 'provider-b' }
const C = { model: 'model-c', provider: 'provider-c' }

const catalog: ModelOptionsResult = {
  ...A,
  providers: [A, B, C].map(row => ({ models: [row.model], name: row.provider, slug: row.provider }))
}

function mountLivePicker() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const protocol = new CanonicalDesktopProtocol()
  const owner = { effort: 'ultra', generation: 1, model: A.model, provider: A.provider, revision: 1 }

  const mutations: Array<{
    expected_generation: unknown
    expected_revision: unknown
    operation: unknown
    payload: unknown
  }> = []

  protocol.result(
    'session.resume',
    { session_id: SID },
    {
      execution_generation: owner.generation,
      messages: [],
      prompts: [],
      revision: owner.revision,
      running: false,
      session_id: SID
    }
  )

  const pendingModelReplies: Array<ReturnType<typeof deferred<Record<string, unknown>>>> = []

  const requestGateway = async <T,>(method: string, params?: Record<string, unknown>): Promise<T> => {
    if (method === 'model.options') {
      return catalog as T
    }

    const prepared = protocol.prepare(method, params ?? {}) as Record<string, unknown>

    if (prepared.operation === 'model' || prepared.operation === 'reasoning') {
      mutations.push({
        expected_generation: prepared.expected_generation,
        expected_revision: prepared.expected_revision,
        operation: prepared.operation,
        payload: prepared.payload
      })

      if (prepared.expected_revision !== owner.revision) {
        throw new Error('revision_conflict')
      }

      if (prepared.expected_generation !== owner.generation) {
        throw new Error('stale_generation')
      }

      if (prepared.operation === 'model') {
        const reply = deferred<Record<string, unknown>>()
        pendingModelReplies.push(reply)
        const response = await reply.promise

        if (!('status' in response)) {
          const payload = prepared.payload as { model: string; provider: string }
          owner.model = payload.model
          owner.provider = payload.provider
          owner.revision += 1
          owner.generation += 1
        }

        const wire = {
          execution_generation: owner.generation,
          model: owner.model,
          operation: 'model',
          provider: owner.provider,
          revision: owner.revision,
          session_id: SID,
          ...response
        }

        return protocol.result(method, prepared, wire) as T
      }

      const payload = prepared.payload as { effort: string }
      owner.effort = payload.effort
      owner.revision += 1
      owner.generation += 1

      return protocol.result(method, prepared, {
        effort: owner.effort,
        execution_generation: owner.generation,
        operation: 'reasoning',
        revision: owner.revision,
        session_id: SID
      }) as T
    }

    throw new Error(`Unexpected RPC: ${method} ${JSON.stringify(params)}`)
  }

  const hook = renderHook(
    () => {
      const cache = useSessionStateCache({
        activeSessionId: SID,
        busyRef: useRef(false),
        selectedStoredSessionId: SID,
        setAwaitingResponse,
        setBusy,
        setMessages
      })

      const controls = useModelControls({ queryClient, requestGateway })
      const menu = useModelMenuController({ onSelectModel: controls.selectModel, requestGateway })

      return { cache, controls, menu }
    },
    { wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider> }
  )

  setSessionTileDelegate({ updateSession: hook.result.current.cache.updateSessionState } as SessionTileDelegate)

  act(() => {
    hook.result.current.cache.updateSessionState(SID, state => ({
      ...state,
      ...sessionInfoStatePatch({
        model: owner.model,
        provider: owner.provider,
        reasoning_effort: owner.effort,
        reasoning_effort_wire: owner.effort
      })
    }))
  })

  const selectWithPreset = async (row: typeof B) => {
    const { controller, defaultEffort } = hook.result.current.menu
    const preset = controller.presetFor(row.provider, row.model)
    const applied = await controller.select(row.model, row.provider)

    if (applied !== false) {
      controller.applyPreset({ effort: preset.effort ?? defaultEffort }, row)
    }

    return applied
  }

  return {
    ...hook,
    mutations,
    owner,
    pendingModelReplies,
    queryClient,
    selectWithPreset
  }
}

beforeEach(() => {
  $activeGatewayProfile.set('default')
  $activeSessionId.set(SID)
  $defaultReasoningEffort.set('medium')
  $modelPresets.set({})
  $sessionStates.set({})
  setCurrentModel(A.model)
  setCurrentProvider(A.provider)
  setCurrentReasoningEffort('ultra')
})

afterEach(() => {
  cleanup()
  setSessionTileDelegate(null as unknown as SessionTileDelegate)
  $activeSessionId.set(null)
  $sessionStates.set({})
  $modelPresets.set({})
  $defaultReasoningEffort.set('')
})

describe('live primary model pick repaint and follow-up preset fencing', () => {
  it('repaints the primary live slice immediately and applies the follow-up effort preset with the adopted generation', async () => {
    setModelPreset(B.provider, B.model, { effort: 'medium' })
    const h = mountLivePicker()

    let pick!: Promise<boolean | void>

    act(() => {
      pick = h.selectWithPreset(B)
    })

    expect(PRIMARY_SESSION_VIEW.$model.get()).toBe(B.model)
    expect(PRIMARY_SESSION_VIEW.$provider.get()).toBe(B.provider)

    await act(async () => {
      h.pendingModelReplies[0].resolve({})
      await expect(pick).resolves.toBe(true)
    })

    await waitFor(() =>
      expect(h.mutations).toEqual([
        {
          expected_generation: 1,
          expected_revision: 1,
          operation: 'model',
          payload: { model: B.model, provider: B.provider }
        },
        {
          expected_generation: 2,
          expected_revision: 2,
          operation: 'reasoning',
          payload: { effort: 'medium' }
        }
      ])
    )

    expect(h.owner).toMatchObject({
      effort: 'medium',
      generation: 3,
      model: B.model,
      provider: B.provider,
      revision: 3
    })
    h.queryClient.clear()
  })

  it('drops an older pick preset when a newer pick replaces the primary live slice before it settles', async () => {
    setModelPreset(B.provider, B.model, { effort: 'low' })
    setModelPreset(C.provider, C.model, { effort: 'high' })
    const h = mountLivePicker()

    let older!: Promise<boolean | void>
    let newer!: Promise<boolean | void>

    act(() => {
      older = h.selectWithPreset(B)
    })

    expect(PRIMARY_SESSION_VIEW.$model.get()).toBe(B.model)

    await act(async () => {
      h.pendingModelReplies[0].resolve({})
      newer = h.selectWithPreset(C)
      await expect(older).resolves.toBe(true)
    })

    expect(PRIMARY_SESSION_VIEW.$model.get()).toBe(C.model)

    await act(async () => {
      h.pendingModelReplies[1].resolve({})
      await expect(newer).resolves.toBe(true)
    })

    await waitFor(() => expect(h.owner.effort).toBe('high'))
    expect(h.mutations.filter(m => m.operation === 'reasoning').map(m => m.payload)).toEqual([{ effort: 'high' }])
    expect(PRIMARY_SESSION_VIEW.$model.get()).toBe(C.model)
    expect(PRIMARY_SESSION_VIEW.$provider.get()).toBe(C.provider)
    h.queryClient.clear()
  })
})

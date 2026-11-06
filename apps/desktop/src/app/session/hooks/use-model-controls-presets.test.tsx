import type { ModelOptionsResult } from '@hermes/shared'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, renderHook } from '@testing-library/react'
import { useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { PRIMARY_SESSION_VIEW } from '@/app/chat/session-view'
import { useModelMenuController } from '@/app/shell/use-model-menu-controller'
import { confirm } from '@/store/confirm'
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
vi.mock('@/store/confirm', () => ({ confirm: vi.fn() }))

const SID = 'primary-preset-order'
const A = { model: 'model-a', provider: 'provider-a' }
const B = { model: 'model-b', provider: 'provider-b' }
const C = { model: 'model-c', provider: 'provider-c' }

const catalog: ModelOptionsResult = {
  ...A,
  providers: [A, B, C].map(row => ({ models: [row.model], name: row.provider, slug: row.provider }))
}

function mountSelection() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const replies = [deferred<object>(), deferred<object>()]
  const reasoningWrites: Array<Record<string, unknown>> = []
  let modelRequest = 0
  let runtime = { ...A, effort: 'ultra' }

  const request = async <T,>(method: string, params?: Record<string, unknown>): Promise<T> => {
    if (method === 'model.options') {
      return catalog as T
    }

    if (method === 'config.set' && params?.key === 'model') {
      return (await replies[modelRequest++].promise) as T
    }

    if (method === 'config.set' && params?.key === 'reasoning') {
      reasoningWrites.push(params)
      runtime.effort = String(params.value)

      return {} as T
    }

    throw new Error(`Unexpected RPC: ${method} ${JSON.stringify(params)}`)
  }

  const hook = renderHook(
    () => {
      const cache = useSessionStateCache({
        activeSessionId: SID,
        selectedStoredSessionId: SID,
        busyRef: useRef(false),
        setAwaitingResponse,
        setBusy,
        setMessages
      })

      const controls = useModelControls({ queryClient, requestGateway: request })
      const menu = useModelMenuController({ onSelectModel: controls.selectModel, requestGateway: request })

      return { cache, controls, menu }
    },
    { wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider> }
  )

  // The shell's delegate forwards to this same cache in production. Keep its
  // unrelated UI actions inert; the state update and publication are real.
  setSessionTileDelegate({ updateSession: hook.result.current.cache.updateSessionState } as SessionTileDelegate)

  const info = (report = runtime) => {
    hook.result.current.cache.updateSessionState(SID, state => ({
      ...state,
      ...sessionInfoStatePatch({
        model: report.model,
        provider: report.provider,
        reasoning_effort: report.effort,
        reasoning_effort_wire: report.effort
      })
    }))
  }

  act(info)

  // Match selectFamily's select -> remembered preset/default ordering, using
  // the real controller and preset adapter rather than a mocked applyPreset.
  const select = async (row: typeof B) => {
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
    info,
    queryClient,
    reasoningWrites,
    select,
    accept: (index: number, row: typeof B, response: object = {}) => {
      if (!('deferred' in response)) {
        runtime = { ...runtime, ...row }
      }

      replies[index].resolve(response)
    },
    runtime: () => runtime
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
  $activeSessionId.set(null)
  $sessionStates.set({})
  $modelPresets.set({})
  $defaultReasoningEffort.set('')
})

describe('primary model selection and reasoning preset ordering', () => {
  it.each([
    { preset: 'low', expected: 'low' },
    { preset: undefined, expected: 'medium' }
  ])('applies $expected after the model RPC, before session.info', async ({ preset, expected }) => {
    if (preset) {
      setModelPreset(B.provider, B.model, { effort: preset })
    }

    const h = mountSelection()
    let pick!: Promise<boolean | void>
    act(() => {
      pick = h.select(B)
    })
    await act(async () => {
      h.accept(0, B)
      await expect(pick).resolves.toBe(true)
    })

    expect(h.reasoningWrites).toEqual([{ key: 'reasoning', session_id: SID, value: expected }])
    act(h.info)
    expect(PRIMARY_SESSION_VIEW.$model.get()).toBe(B.model)
    expect(PRIMARY_SESSION_VIEW.$provider.get()).toBe(B.provider)
    expect(PRIMARY_SESSION_VIEW.$reasoningEffort.get()).toBe(expected)
    expect(h.runtime()).toEqual({ ...B, effort: expected })
    h.queryClient.clear()
  })

  it('preserves authoritative info delivered before the model RPC', async () => {
    setModelPreset(B.provider, B.model, { effort: 'low' })
    const h = mountSelection()
    let pick!: Promise<boolean | void>
    act(() => {
      pick = h.select(B)
    })
    await act(async () => {
      h.accept(0, B)
      h.info()
      expect(PRIMARY_SESSION_VIEW.$reasoningEffortWire.get()).toBe('ultra')
      await pick
    })
    expect(h.reasoningWrites).toEqual([{ key: 'reasoning', session_id: SID, value: 'low' }])
    act(h.info)
    expect(PRIMARY_SESSION_VIEW.$reasoningEffort.get()).toBe('low')
    h.queryClient.clear()
  })

  it('rejects the older preset when a genuinely newer selection already applied', async () => {
    setModelPreset(B.provider, B.model, { effort: 'low' })
    setModelPreset(C.provider, C.model, { effort: 'high' })
    const h = mountSelection()
    let older!: Promise<boolean | void>
    let newer!: Promise<boolean | void>
    act(() => {
      older = h.select(B)
      newer = h.select(C)
    })
    await act(async () => {
      h.accept(1, C)
      h.info()
      await newer
    })
    await act(async () => {
      // An old reply is not another runtime switch; C remains authoritative.
      h.accept(0, C)
      await older
    })
    expect(h.reasoningWrites).toEqual([{ key: 'reasoning', session_id: SID, value: 'high' }])
    act(h.info)
    expect(PRIMARY_SESSION_VIEW.$model.get()).toBe(C.model)
    expect(PRIMARY_SESSION_VIEW.$reasoningEffort.get()).toBe('high')
    h.queryClient.clear()
  })

  it('keeps the wire stamp when session.info already reported the accepted route', async () => {
    const h = mountSelection()
    let pick!: Promise<boolean>
    act(() => {
      pick = h.result.current.controls.selectModel(B)
    })
    await act(async () => {
      h.accept(0, B)
      h.info()
      await pick
    })
    expect(PRIMARY_SESSION_VIEW.$model.get()).toBe(B.model)
    expect(PRIMARY_SESSION_VIEW.$reasoningEffortWire.get()).toBe('ultra')
    h.queryClient.clear()
  })

  it('does not publish an older acceptance while a newer model pick is pending', async () => {
    setModelPreset(B.provider, B.model, { effort: 'low' })
    setModelPreset(C.provider, C.model, { effort: 'high' })
    const h = mountSelection()
    let older!: Promise<boolean | void>
    let newer!: Promise<boolean | void>
    act(() => {
      older = h.select(B)
      newer = h.select(C)
    })
    await act(async () => {
      h.accept(0, B)
      await older
    })
    expect(PRIMARY_SESSION_VIEW.$model.get()).toBe(A.model)
    expect(h.reasoningWrites).toEqual([])
    await act(async () => {
      h.accept(1, C)
      await newer
    })
    expect(h.reasoningWrites).toEqual([{ key: 'reasoning', session_id: SID, value: 'high' }])
    act(h.info)
    expect(PRIMARY_SESSION_VIEW.$model.get()).toBe(C.model)
    expect(PRIMARY_SESSION_VIEW.$reasoningEffort.get()).toBe('high')
    h.queryClient.clear()
  })

  it('publishes a confirmed retry through the same live state path before info', async () => {
    const confirmation = deferred<boolean>()
    vi.mocked(confirm).mockReturnValueOnce(confirmation.promise)
    const h = mountSelection()
    let pick!: Promise<boolean>
    act(() => {
      pick = h.result.current.controls.selectModel(B)
    })
    await act(async () => {
      h.accept(0, A, { confirm_required: true })
      await expect(pick).resolves.toBe(false)
    })
    await act(async () => {
      confirmation.resolve(true)
    })
    await act(async () => {
      h.accept(1, B)
    })
    expect(PRIMARY_SESSION_VIEW.$model.get()).toBe(B.model)
    expect(PRIMARY_SESSION_VIEW.$provider.get()).toBe(B.provider)
    expect(PRIMARY_SESSION_VIEW.$reasoningEffortWire.get()).toBe('')
    act(h.info)
    expect(PRIMARY_SESSION_VIEW.$reasoningEffortWire.get()).toBe('ultra')
    h.queryClient.clear()
  })

  it('still accepts authoritative session.info after reconciling a model reply', async () => {
    setModelPreset(B.provider, B.model, { effort: 'low' })
    const h = mountSelection()
    let pick!: Promise<boolean | void>
    act(() => {
      pick = h.select(B)
    })
    await act(async () => {
      h.accept(0, B)
      await pick
    })
    act(() => {
      h.info({ ...C, effort: 'ultra' })
    })
    expect(PRIMARY_SESSION_VIEW.$model.get()).toBe(C.model)
    expect(PRIMARY_SESSION_VIEW.$provider.get()).toBe(C.provider)
    expect(h.reasoningWrites).toEqual([{ key: 'reasoning', session_id: SID, value: 'low' }])
    expect(PRIMARY_SESSION_VIEW.$reasoningEffort.get()).toBe('ultra')
    h.queryClient.clear()
  })

  it('applies the newest preset when an older pick reports info during its request', async () => {
    setModelPreset(B.provider, B.model, { effort: 'low' })
    setModelPreset(C.provider, C.model, { effort: 'high' })
    const h = mountSelection()
    let older!: Promise<boolean | void>
    let newer!: Promise<boolean | void>
    act(() => {
      older = h.select(B)
      newer = h.select(C)
    })
    await act(async () => {
      h.accept(0, B)
      h.info()
      await older
    })
    expect(h.reasoningWrites).toEqual([{ key: 'reasoning', session_id: SID, value: 'low' }])
    await act(async () => {
      h.accept(1, C)
      await newer
    })
    expect(h.reasoningWrites).toEqual([
      { key: 'reasoning', session_id: SID, value: 'low' },
      { key: 'reasoning', session_id: SID, value: 'high' }
    ])
    act(h.info)
    expect(PRIMARY_SESSION_VIEW.$model.get()).toBe(C.model)
    expect(PRIMARY_SESSION_VIEW.$reasoningEffort.get()).toBe('high')
    h.queryClient.clear()
  })

  it('keeps a deferred selection off the active route and does not send its preset', async () => {
    setModelPreset(B.provider, B.model, { effort: 'low' })
    const h = mountSelection()
    let pick!: Promise<boolean | void>
    act(() => {
      pick = h.select(B)
    })
    await act(async () => {
      h.accept(0, B, { deferred: true })
      await expect(pick).resolves.toBe(true)
    })
    expect(PRIMARY_SESSION_VIEW.$model.get()).toBe(A.model)
    expect(PRIMARY_SESSION_VIEW.$provider.get()).toBe(A.provider)
    expect(h.reasoningWrites).toEqual([])
    expect(h.runtime()).toEqual({ ...A, effort: 'ultra' })
    act(h.info)
    expect(PRIMARY_SESSION_VIEW.$reasoningEffort.get()).toBe('ultra')
    h.queryClient.clear()
  })
})

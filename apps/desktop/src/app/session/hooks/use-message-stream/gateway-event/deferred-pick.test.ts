import { afterEach, describe, expect, it, vi } from 'vitest'

import { $currentModel, $currentProvider, setCurrentModelTransient, setCurrentProviderTransient } from '@/store/session'

import { handleDeferredPickEvent } from './deferred-pick'
import type { GatewayEventContext } from './types'

const notifyError = vi.hoisted(() => vi.fn())
const surfaceModelSwitchConfirm = vi.hoisted(() => vi.fn(async () => true))
const requestGatewayForAgent = vi.hoisted(() => vi.fn(async () => ({})))

vi.mock('@/store/notifications', async importOriginal => ({ ...(await importOriginal<object>()), notifyError }))
vi.mock('@/lib/guarded-model-switch', () => ({ surfaceModelSwitchConfirm }))
vi.mock('@/store/gateway', async importOriginal => ({ ...(await importOriginal<object>()), requestGatewayForAgent }))

function context(payload: Record<string, unknown>, overrides: Record<string, unknown> = {}) {
  const states: Record<string, Record<string, unknown>> = { s: { model: 'picked', provider: 'custom' } }

  const ctx = {
    deps: {
      activeGatewayProfile: 'default',
      activeSessionIdRef: { current: 's' },
      queryClient: { invalidateQueries: vi.fn(async () => undefined) },
      updateSessionState: (sid: string, updater: (state: never) => never) => (states[sid] = updater(states[sid] as never))
    },
    event: { payload, session_id: 's', type: 'session.updated' },
    explicitSid: 's',
    fromActiveSource: () => true,
    isActiveEvent: true,
    occurredAt: 0,
    payload,
    scheduleConfigRefresh: vi.fn(),
    sessionId: 's',
    ...overrides
  } as unknown as GatewayEventContext

  return { ctx, states }
}

describe('handleDeferredPickEvent', () => {
  afterEach(() => { vi.clearAllMocks() })

  it('shows a refused next-turn pick by name and repaints the committed model', () => {
    setCurrentModelTransient('picked')
    setCurrentProviderTransient('custom')

    const { ctx, states } = context({ model: 'frozen', operation: 'model', provider: 'openrouter',
      reason: 'model_resolution_failed', session_id: 's', status: 'refused', target_model: 'picked' })

    expect(handleDeferredPickEvent(ctx)).toBe(true)
    expect(notifyError).toHaveBeenCalledWith(expect.objectContaining({ message: 'model_resolution_failed' }), 'Model switch failed')
    expect(states.s).toMatchObject({ model: 'frozen', provider: 'openrouter' })
    expect([$currentModel.get(), $currentProvider.get()]).toEqual(['frozen', 'openrouter'])
  })

  it('asks for a guarded next-turn pick instead of dropping it', () => {
    const { ctx } = context({ confirm: 'token', confirm_message: 'Costly', operation: 'model',
      payload: { model: 'guarded', provider: 'custom' }, session_id: 's', status: 'confirmation_required', target_model: 'guarded' })

    expect(handleDeferredPickEvent(ctx)).toBe(true)
    expect(surfaceModelSwitchConfirm).toHaveBeenCalledWith(expect.objectContaining({ confirmMessage: 'Costly', model: 'guarded' }))
    expect(notifyError).not.toHaveBeenCalled()
  })

  it('leaves other session updates to their handlers', () => {
    expect(handleDeferredPickEvent(context({ operation: 'rename', session_id: 's', title: 'x' }).ctx)).toBe(false)
    expect(handleDeferredPickEvent(context({ model: 'm', operation: 'model', revision: 3, session_id: 's' }).ctx)).toBe(false)
  })

  it('declining a guarded next-turn pick paints the committed model again', async () => {
    setCurrentModelTransient('guarded')
    setCurrentProviderTransient('custom')
    surfaceModelSwitchConfirm.mockResolvedValueOnce(false)

    const { ctx, states } = context({ committed_model: 'frozen', committed_provider: 'openrouter', confirm: 'token',
      operation: 'model', payload: { model: 'guarded', provider: 'custom' }, session_id: 's',
      status: 'confirmation_required', target_model: 'guarded' })

    states.s = { model: 'guarded', provider: 'custom' }
    expect(handleDeferredPickEvent(ctx)).toBe(true)
    await vi.waitFor(() => expect($currentModel.get()).toBe('frozen'))
    expect($currentProvider.get()).toBe('openrouter')
    expect(states.s).toMatchObject({ model: 'frozen', provider: 'openrouter' })
  })

  it('a decline never rewinds a newer pick or another session', async () => {
    setCurrentModelTransient('guarded')
    let answer: (value: boolean) => void = () => undefined
    surfaceModelSwitchConfirm.mockReturnValueOnce(new Promise<boolean>(resolve => (answer = resolve)))

    const { ctx, states } = context({ committed_model: 'frozen', confirm: 'token', operation: 'model',
      payload: { model: 'guarded' }, session_id: 's', status: 'confirmation_required', target_model: 'guarded' })

    handleDeferredPickEvent(ctx)
    setCurrentModelTransient('newer')
    states.s = { model: 'newer', provider: 'custom' }
    answer(false)
    await new Promise(resolve => setTimeout(resolve, 0))
    expect($currentModel.get()).toBe('newer')
    expect(states.s).toMatchObject({ model: 'newer', provider: 'custom' })
  })

  it('a confirmed request that goes stale in flight never rewinds the newer pick', async () => {
    setCurrentModelTransient('guarded')
    let finishRequest: () => void = () => undefined
    requestGatewayForAgent.mockReturnValueOnce(new Promise(resolve => (finishRequest = () => resolve({}))))
    surfaceModelSwitchConfirm.mockImplementationOnce((async (options: { requestConfirmed: () => Promise<unknown> }) => {
      await options.requestConfirmed()

      return false
    }) as never)

    const { ctx, states } = context({ committed_model: 'frozen', confirm: 'token', operation: 'model',
      payload: { model: 'guarded' }, session_id: 's', status: 'confirmation_required', target_model: 'guarded' })

    handleDeferredPickEvent(ctx)
    await vi.waitFor(() => expect(requestGatewayForAgent).toHaveBeenCalled())
    setCurrentModelTransient('newer')
    states.s = { model: 'newer', provider: 'custom' }
    finishRequest()
    await new Promise(resolve => setTimeout(resolve, 0))
    expect($currentModel.get()).toBe('newer')
    expect(states.s).toMatchObject({ model: 'newer', provider: 'custom' })
  })

  it('a decline never rewinds the same model picked through another provider', async () => {
    setCurrentModelTransient('guarded')
    setCurrentProviderTransient('custom')
    let answer: (value: boolean) => void = () => undefined
    surfaceModelSwitchConfirm.mockReturnValueOnce(new Promise<boolean>(resolve => (answer = resolve)))

    const { ctx, states } = context({ committed_model: 'frozen', committed_provider: 'openrouter', confirm: 'token', operation: 'model',
      payload: { model: 'guarded', provider: 'custom' }, session_id: 's', status: 'confirmation_required', target_model: 'guarded' })

    handleDeferredPickEvent(ctx)
    setCurrentProviderTransient('other')
    states.s = { model: 'guarded', provider: 'other' }
    answer(false)
    await new Promise(resolve => setTimeout(resolve, 0))
    expect([$currentModel.get(), $currentProvider.get()]).toEqual(['guarded', 'other'])
    expect(states.s).toMatchObject({ model: 'guarded', provider: 'other' })
  })

  it('a settlement after the active source moved never repaints a reused runtime', async () => {
    setCurrentModelTransient('guarded')
    setCurrentProviderTransient('custom')
    let fromActive = true
    let answer: (value: boolean) => void = () => undefined
    surfaceModelSwitchConfirm.mockReturnValueOnce(new Promise<boolean>(resolve => (answer = resolve)))

    const { ctx, states } = context({ committed_model: 'frozen', committed_provider: 'openrouter', confirm: 'token', operation: 'model',
      payload: { model: 'guarded', provider: 'custom' }, session_id: 's', status: 'confirmation_required', target_model: 'guarded' },
    { fromActiveSource: () => fromActive })

    states.s = { model: 'guarded', provider: 'custom' }
    handleDeferredPickEvent(ctx)
    fromActive = false
    answer(false)
    await new Promise(resolve => setTimeout(resolve, 0))
    expect([$currentModel.get(), $currentProvider.get()]).toEqual(['guarded', 'custom'])
    expect(states.s).toMatchObject({ model: 'guarded', provider: 'custom' })
  })

  it("another profile's same-ID refusal leaves the active composer and its cache alone", () => {
    setCurrentModelTransient('picked')
    setCurrentProviderTransient('custom')

    const { ctx, states } = context({ model: 'other-profile-model', operation: 'model', provider: 'openrouter',
      reason: 'model_resolution_failed', session_id: 's', status: 'refused', target_model: 'picked' },
    { event: { payload: {}, profile: 'other', session_id: 's', type: 'session.updated' }, fromActiveSource: () => false })

    expect(handleDeferredPickEvent(ctx)).toBe(true)
    expect([$currentModel.get(), $currentProvider.get()]).toEqual(['picked', 'custom'])
    expect(states.s).toMatchObject({ model: 'picked', provider: 'custom' })
    expect((ctx.deps.queryClient.invalidateQueries as any).mock.calls[0][0].queryKey).toContain('other')
  })

  it('confirming a combined model + reasoning pick resends the whole selection with the owner token', async () => {
    surfaceModelSwitchConfirm.mockImplementationOnce((async (options: { requestConfirmed: () => Promise<unknown> }) => {
      await options.requestConfirmed()

      return true
    }) as never)

    const { ctx } = context({ confirm: 'token', operation: 'model', payload: { model: 'guarded', provider: 'custom', reasoning: 'high' },
      session_id: 's', status: 'confirmation_required', target_model: 'guarded' })

    handleDeferredPickEvent(ctx)
    await vi.waitFor(() => expect(requestGatewayForAgent).toHaveBeenCalled())
    expect(requestGatewayForAgent).toHaveBeenCalledWith(null, 'default', 'session.model', {
      confirm: 'token', payload: { model: 'guarded', provider: 'custom', reasoning: 'high' }, session_id: 's' })
  })
})

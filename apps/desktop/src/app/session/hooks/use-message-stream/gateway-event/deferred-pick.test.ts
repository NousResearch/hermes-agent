import type { GatewayEvent } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import type * as GuardedModelSwitch from '@/lib/guarded-model-switch'
import { $confirmRequest, settleConfirm } from '@/store/confirm'
import { $currentModel, $currentProvider, clearComposerSelectionOwner, setComposerSelectionOwner, setCurrentModelTransient, setCurrentProviderTransient } from '@/store/session'
import { $sessionTiles, clearAllSessionStates, recordSessionEventScope } from '@/store/session-states'

import { renderMessageStream } from '../test-harness'

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

    const { ctx } = context({ committed_model: 'frozen', confirm: 'token', operation: 'model',
      payload: { model: 'guarded' }, session_id: 's', status: 'confirmation_required', target_model: 'guarded' })

    handleDeferredPickEvent(ctx)
    setCurrentModelTransient('newer')
    answer(false)
    await new Promise(resolve => setTimeout(resolve, 0))
    expect($currentModel.get()).toBe('newer')
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

// These receipts arrive on a tile's own registry socket while another chat stays visible.
// Use the production fan-in, stream dispatcher and confirm decision; only the owner RPC is stubbed.
describe('background model receipts', () => {
  const id = 'tile-runtime'
  const owner = { connectionId: 'tile-owner', profile: 'default' }
  const beforeTiles = $sessionTiles.get()
  const committed = { model: 'tile-old', provider: 'tile-provider' }
  const requested = { model: 'tile-next', provider: 'tile-provider' }

  function mountTile() {
    const active = { current: 'foreground-a' }
    setComposerSelectionOwner('foreground-owner', 'default')
    setCurrentModelTransient('foreground-model')
    setCurrentProviderTransient('foreground-provider')
    $sessionTiles.set([{ ownerRoute: owner, runtimeId: id, storedSessionId: id }])

    const states = new Map([[id, { ...createClientSessionState(), ...committed, modelRevision: 3,
      queuedModel: { ...requested, revision: 3 } }]])

    const stream = renderMessageStream(active.current, { activeSessionIdRef: active, states })

    const deliver = (fields: Record<string, unknown>, connectionId = owner.connectionId) => {
      const frame = { ...owner, connectionId, type: 'session.updated', session_id: id,
        payload: { operation: 'model', session_id: id, ...fields } } as unknown as GatewayEvent

      recordSessionEventScope(frame)
      act(() => stream.handleEvent(frame))
    }

    const info = (connectionId: string, model: string, revision: number) => {
      const frame = { ...owner, connectionId, type: 'session.info', session_id: id,
        payload: { model, provider: requested.provider, revision } } as unknown as GatewayEvent

      recordSessionEventScope(frame)
      act(() => stream.handleEvent(frame))
    }

    const ask = () => deliver({ status: 'confirmation_required', confirm: 'tile-fixture',
      target_model: requested.model, payload: requested, committed_model: committed.model,
      committed_provider: committed.provider })

    return { active, ask, deliver, info, read: () => stream.state(id) }
  }

  afterEach(() => {
    cleanup()
    settleConfirm(false)
    clearAllSessionStates()
    $sessionTiles.set(beforeTiles)
    clearComposerSelectionOwner()
    vi.clearAllMocks()
  })

  it('publishes the owning background commit without waiting for session.info or changing the foreground', () => {
    const { deliver, read } = mountTile()
    deliver({ ...requested, revision: 4, status: 'applied' })
    expect(read()).toMatchObject({ ...requested, modelRevision: 4, queuedModel: undefined })
    expect([$currentModel.get(), $currentProvider.get()]).toEqual(['foreground-model', 'foreground-provider'])
  })

  it('keeps a same-id background tile on its bound owner when another connection also reports a commit', () => {
    const { deliver, read } = mountTile()
    deliver({ model: 'other-model', provider: 'other-provider', revision: 99, status: 'applied' }, 'other-owner')
    expect(read()).toMatchObject({ ...committed, modelRevision: 3, queuedModel: requested })
  })

  it('only the bound tile owner can settle its queued label through session.info', () => {
    const { info, read } = mountTile()
    info('other-owner', requested.model, 99)
    expect(read()).toMatchObject({ ...committed, modelRevision: 3, queuedModel: requested })
    info(owner.connectionId, requested.model, 4)
    expect(read()).toMatchObject({ ...requested, modelRevision: 4, queuedModel: undefined })
  })

  it('confirms on the background owner after unrelated foreground navigation and publishes its receipt', async () => {
    const { surfaceModelSwitchConfirm: actualConfirm } = await vi.importActual<typeof GuardedModelSwitch>('@/lib/guarded-model-switch')
    surfaceModelSwitchConfirm.mockImplementationOnce(actualConfirm as never)
    requestGatewayForAgent.mockResolvedValueOnce({ ...requested, revision: 4 })
    const { active, ask, read } = mountTile()
    ask()
    expect($confirmRequest.get()).not.toBeNull()
    active.current = 'foreground-c'
    await act(async () => { settleConfirm(true) })

    expect(requestGatewayForAgent).toHaveBeenCalledExactlyOnceWith(owner.connectionId, owner.profile, 'session.model', {
      session_id: id, confirm: 'tile-fixture', payload: requested
    })
    expect(read()).toMatchObject({ ...requested, modelRevision: 4, queuedModel: undefined })
    expect($currentModel.get()).toBe('foreground-model')
  })

  it('does not confirm an outdated background choice after that tile receives a newer committed route', async () => {
    const { surfaceModelSwitchConfirm: actualConfirm } = await vi.importActual<typeof GuardedModelSwitch>('@/lib/guarded-model-switch')
    surfaceModelSwitchConfirm.mockImplementationOnce(actualConfirm as never)
    const { ask, deliver, read } = mountTile()
    ask()
    expect($confirmRequest.get()).not.toBeNull()
    deliver({ model: 'newer-tile-model', provider: 'tile-provider', revision: 5, status: 'applied' })
    await act(async () => { settleConfirm(true) })
    expect(requestGatewayForAgent).not.toHaveBeenCalled()
    expect(read().model).toBe('newer-tile-model')
  })
})

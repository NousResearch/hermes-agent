import { QueryClient } from '@tanstack/react-query'
import { cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $activeSessionId,
  $currentProvider,
  $currentReasoningEffortWire,
  clearComposerSelectionOwner,
  markComposerSelectionManual,
  restoreComposerModelSelection,
  setComposerSelectionOwner,
  setConnection,
  setCurrentModel,
  setCurrentModelSource,
  setCurrentProvider,
  setCurrentProviderTransient,
  setCurrentReasoningEffortWire
} from '@/store/session'

import { deferred } from '../../../../test/deferred'
import { handleDeferredPickEvent } from '../use-message-stream/gateway-event/deferred-pick'
import type { GatewayEventContext } from '../use-message-stream/gateway-event/types'
import { useModelControls } from '../use-model-controls'

import { desktopSessionCreateParams } from './session-create-request'
import { applyRuntimeInfo, applyStoredSessionPreviewRuntimeInfo } from './utils'

const confirm = vi.hoisted(() => vi.fn())

vi.mock('@/lib/guarded-model-switch', () => ({ surfaceModelSwitchConfirm: confirm }))
vi.mock('@/store/notifications', () => ({ dismissNotification: vi.fn(), notify: vi.fn(), notifyError: vi.fn() }))
vi.mock('@/store/profile', async original => ({
  ...(await original<Record<string, unknown>>()),
  ensureGatewayProfile: vi.fn(async () => undefined)
}))

const picked = { model: 'grok-4.7', provider: 'xai-oauth' }
const launched = async () => (await desktopSessionCreateParams('')).model

const pick = (reply: unknown, model = 'queued', sessionId = 'busy') =>
  renderHook(() => useModelControls({ queryClient: new QueryClient(), requestGateway: vi.fn(async () => reply) as never }))
    .result.current.selectModel({ model, provider: 'p', sessionId })

// The owner's verdict on a pick queued for the next turn.
function verdict(payload: Record<string, unknown>, session = 'busy', viewing = 'busy') {
  const full = { operation: 'model', session_id: session, target_model: 'queued', ...payload }
  handleDeferredPickEvent({
    deps: {
      activeGatewayProfile: 'default',
      activeSessionIdRef: { current: viewing },
      queryClient: { invalidateQueries: vi.fn(async () => undefined) },
      updateSessionState: vi.fn()
    },
    event: { payload: full, session_id: session, type: 'session.updated' },
    fromActiveSource: () => true,
    payload: full,
    sessionId: session
  } as unknown as GatewayEventContext)
}

const refuse = (session?: string, viewing?: string) => verdict({ ...picked, status: 'refused' }, session, viewing)

beforeEach(() => {
  setConnection({ mode: 'local', baseUrl: '' } as never)
  $activeSessionId.set('busy')
  setCurrentModel(picked.model)
  setCurrentProvider(picked.provider)
  markComposerSelectionManual()
})

afterEach(() => {
  cleanup()
  clearComposerSelectionOwner()
  $activeSessionId.set(null)
  setCurrentModelSource('')
  setConnection(null)
  window.localStorage.clear()
})

describe('new chats launch on the picked route', () => {
  // #143323: a fallback to another provider serving the same model id fails nothing and bills the wrong
  // account; a named provider's runtime reports the bare `custom` class, which cannot resolve on create.
  it.each(['opencode-go', 'custom'])('not the runtime the pill shows (%s)', async provider => {
    applyRuntimeInfo({ model: 'runtime-model', provider })
    await expect(desktopSessionCreateParams('')).resolves.toMatchObject(picked)
    setCurrentProviderTransient(provider)
    applyStoredSessionPreviewRuntimeInfo({ model: 'runtime-model' }, 'abandoned-resume')
    await expect(desktopSessionCreateParams('')).resolves.toMatchObject(picked)
  })

  it('after a refused switch, which restores the pick and the runtime display separately', async () => {
    setCurrentProviderTransient('opencode-go')
    await expect(pick(Promise.reject(new Error('refused')))).resolves.toBe(false)
    await expect(desktopSessionCreateParams('')).resolves.toMatchObject(picked)
    expect($currentProvider.get()).toBe('opencode-go')
  })

  it.each([
    { switched: false, launches: undefined },
    { switched: true, launches: 'queued' }
  ])('after a switch that fails once its chat is left, not the failed pick (owner switched: $switched)', async ({ switched, launches }) => {
    const failed = deferred<never>()
    const switching = pick(failed.promise)
    $activeSessionId.set('elsewhere')

    if (switched) {
      setComposerSelectionOwner('other-backend', 'other')
      setCurrentModel('queued')
      setCurrentProvider('p')
      markComposerSelectionManual()
    }

    failed.reject(new Error('refused'))
    await expect(switching).resolves.toBe(false)
    expect(await launched()).toBe(launches)
  })

  it('which a fresh draft repaints without the outgoing route\'s effort label', () => {
    setCurrentProviderTransient('opencode-go')
    setCurrentReasoningEffortWire('max')
    restoreComposerModelSelection()
    expect([$currentProvider.get(), $currentReasoningEffortWire.get()]).toEqual([picked.provider, ''])
  })
})

describe('a pick queued for the next turn', () => {
  it.each([
    { when: 'is refused', then: () => refuse(), launches: undefined },
    { when: 'is refused after switching chats', then: () => refuse('busy', 'elsewhere'), launches: undefined },
    { when: 'is overtaken by a newer pick', then: () => (setCurrentModel('newer'), refuse()), launches: 'newer' },
    { when: 'is overtaken by its model on another provider', then: () => (setCurrentProvider('other'), refuse()), launches: 'queued' },
    { when: 'has a same-model tile pick refused', then: async () => (await pick({ deferred: true }, 'queued', 'tile'), refuse('tile')), launches: 'queued' },
    {
      when: 'is followed by a tile pick on its session that is refused',
      then: async () => ($activeSessionId.set('elsewhere'), await pick({ deferred: true }), refuse('busy', 'elsewhere')),
      launches: 'queued'
    }
  ])('$when', async ({ then, launches }) => {
    await pick({ deferred: true })
    await then()
    expect(await launched()).toBe(launches)
  })

  it.each([
    { reloaded: false, switched: false, session: 'busy', launches: undefined },
    { reloaded: true, switched: false, session: 'busy', launches: undefined },
    { reloaded: true, switched: false, session: 'tile', launches: 'queued' },
    { reloaded: false, switched: true, session: 'busy', launches: 'queued' }
  ])('with its cost prompt declined (reloaded: $reloaded, owner switched: $switched, $session)', async ({ reloaded, switched, session, launches }) => {
    const answer = deferred<boolean>()
    confirm.mockReturnValueOnce(answer.promise)
    // After a reload the gateway asks again for a pick this run never queued.
    await (reloaded ? (setCurrentModel('queued'), setCurrentProvider('p')) : pick({ deferred: true }))
    verdict({ committed_model: picked.model, confirm: 'token', payload: { model: 'queued', provider: 'p' },
      status: 'confirmation_required' }, session)

    if (switched) {
      setComposerSelectionOwner('other-backend', 'other')
      setCurrentModel('queued')
      setCurrentProvider('p')
      markComposerSelectionManual()
    }

    answer.resolve(false)
    await answer.promise
    await Promise.resolve()
    expect(await launched()).toBe(launches)
  })
})

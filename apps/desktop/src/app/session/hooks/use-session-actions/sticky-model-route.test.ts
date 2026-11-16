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
function verdict(payload: Record<string, unknown>, session = 'busy', viewing = 'busy', source?: Record<string, string>) {
  const full = { operation: 'model', session_id: session, target_model: 'queued', ...payload }
  handleDeferredPickEvent({
    deps: {
      activeGatewayProfile: 'default',
      activeSessionIdRef: { current: viewing },
      queryClient: { invalidateQueries: vi.fn(async () => undefined) },
      updateSessionState: vi.fn()
    },
    event: { payload: full, session_id: session, type: 'session.updated', ...source },
    fromActiveSource: () => !source,
    payload: full,
    sessionId: session
  } as unknown as GatewayEventContext)
}

const refuse = (session?: string, viewing?: string) => verdict({ ...picked, status: 'refused' }, session, viewing)

// Owner B, with its own manual pick of the same route; `backToA` returns to the local owner the tests start on.
function switchToOwnerB() {
  setComposerSelectionOwner('other-backend', 'other')
  setCurrentModel('queued')
  setCurrentProvider('p')
  markComposerSelectionManual()
}

const backToA = () => setConnection({ mode: 'local', baseUrl: '' } as never)

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
      switchToOwnerB()
    }

    failed.reject(new Error('refused'))
    await expect(switching).resolves.toBe(false)
    expect(await launched()).toBe(launches)

    if (switched) {
      // The failed pick was the first owner's: it is released there, not left to launch its new chats.
      backToA()
      expect(await launched()).toBeUndefined()
    }
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
    { when: "is replaced by another client's pick that is refused", then: () => verdict({ status: 'refused', target_model: 'other' }), launches: 'queued' },
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
    { by: 'its chat', switched: false, launches: undefined },
    { by: 'its chat', switched: true, launches: 'queued' },
    // The same route queued from a tile whose chat is then opened in the main pane: the tile never saved it.
    { by: 'a tile', switched: false, launches: 'queued' }
  ])('queued by $by, with its cost prompt declined (owner switched: $switched)', async ({ by, switched, launches }) => {
    const answer = deferred<boolean>()
    confirm.mockReturnValueOnce(answer.promise)
    const session = by === 'a tile' ? 'tile' : 'busy'

    if (by === 'a tile') {
      setCurrentModel('queued')
      setCurrentProvider('p')
    }

    await pick({ deferred: true }, 'queued', session)
    verdict({ committed_model: picked.model, confirm: 'token', payload: { model: 'queued', provider: 'p' },
      status: 'confirmation_required' }, session, session)

    if (switched) {
      switchToOwnerB()
    }

    answer.resolve(false)
    await answer.promise
    await Promise.resolve()
    expect(await launched()).toBe(launches)

    if (switched) {
      backToA()
      expect(await launched()).toBeUndefined()
    }
  })

  it.each([
    { from: 'its owner', source: { connectionId: 'conn-a', profile: 'work' }, launches: undefined },
    // Another owner's session that shares the id (an import) never queued this pick.
    { from: 'another owner', source: { connectionId: 'conn-c', profile: 'work' }, launches: 'queued' }
  ])('that is refused on $from\'s connection while another owner is active', async ({ source, launches }) => {
    setComposerSelectionOwner('conn-a', 'work')
    setCurrentModel(picked.model)
    setCurrentProvider(picked.provider)
    markComposerSelectionManual()
    await pick({ deferred: true })
    switchToOwnerB()
    verdict({ ...picked, status: 'refused' }, 'busy', 'elsewhere', source)
    expect(await launched()).toBe('queued')
    setComposerSelectionOwner('conn-a', 'work')
    expect(await launched()).toBe(launches)
  })

  // Picks a, b, c start in that order and answer as listed; an older late reply never overrides a newer answer.
  it.each([
    { answers: 'b a', refused: 'b' },
    { answers: 'b c a', refused: 'b', fails: 'c' },
    { answers: 'b a c', refused: 'b', fails: 'c' }
  ])('whose replies cross ($answers, $fails fails)', async ({ answers, refused, fails }) => {
    const order = answers.split(' ')
    const replies = Object.fromEntries(order.toSorted().map(model => [model, deferred<unknown>()]))
    const picks = Object.fromEntries(order.toSorted().map(model => [model, pick(replies[model].promise, model)]))

    for (const model of order) {
      if (model === fails) {
        replies[model].reject(new Error('refused'))
      } else {
        replies[model].resolve({ deferred: true })
      }

      await picks[model]
    }

    verdict({ status: 'refused', target_model: refused })
    expect(await launched()).toBeUndefined()
  })

  it('that settles after another owner answers on a session with the same id, and is refused', async () => {
    const reply = deferred<unknown>()
    const switching = pick(reply.promise)
    switchToOwnerB()
    await pick({}, 'other')
    backToA()
    // A failing pick back on this owner rolls back to its own pick still in flight, not the other owner's.
    await pick(Promise.reject(new Error('refused')), 'c')
    reply.resolve({ deferred: true })
    await switching
    refuse()
    expect(await launched()).toBeUndefined()
  })

  it('that settles after an owner switch is recorded for its own owner, and released there when refused', async () => {
    const reply = deferred<unknown>()
    const switching = pick(reply.promise)
    switchToOwnerB()
    reply.resolve({ deferred: true })
    await switching
    // Another chat on owner B, with the same route, is refused: the late result left B's pick alone.
    refuse()
    expect(await launched()).toBe('queued')
    backToA()
    refuse()
    expect(await launched()).toBeUndefined()
  })
})

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom, computed } from 'nanostores'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { PRIMARY_SESSION_VIEW, type SessionView, SessionViewProvider } from '@/app/chat/session-view'
import { applySessionInfoStatePatch, sessionInfoStatePatch } from '@/app/session/hooks/use-message-stream/utils'
import { useModelControls } from '@/app/session/hooks/use-model-controls'
import { desktopSessionCreateParams } from '@/app/session/hooks/use-session-actions/session-create-request'
import type { ClientSessionState } from '@/app/types'
import { DropdownMenu, DropdownMenuContent } from '@/components/ui/dropdown-menu'
import { createClientSessionState } from '@/lib/chat-runtime'
import { confirm } from '@/store/confirm'
import { $customModels } from '@/store/custom-models'
import { $modelPresets, getModelPreset } from '@/store/model-presets'
import { $notifications } from '@/store/notifications'
import { $collapsedProviders } from '@/store/provider-collapse'
import {
  $activeSessionId,
  $currentModel,
  $currentProvider,
  $currentReasoningEffort,
  $selectedStoredSessionId,
  setCurrentFastMode,
  setCurrentModel,
  setCurrentModelSource,
  setCurrentProvider,
  setCurrentReasoningEffort,
  setCurrentServiceTier
} from '@/store/session'
import { $sessionStates } from '@/store/session-states'
import type * as SessionStates from '@/store/session-states'

import { deferred } from '../../test/deferred'

import { ModelMenuCloseContext, ModelMenuPanel } from './model-menu-panel'
import { type ModelMenuHostProps, useModelMenuController } from './use-model-menu-controller'

// Only the network handshake and the wiring cache's store adapter are replaced.
vi.mock('@/store/profile', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureGatewayProfile: vi.fn(async () => undefined),
  ensureGatewayAgent: vi.fn(async () => undefined)
}))
vi.mock('@/store/confirm', () => ({ confirm: vi.fn() }))
vi.mock('@/store/session-states', async importOriginal => {
  const actual = await importOriginal<typeof SessionStates>()

  return {
    ...actual,
    sessionTileDelegate: () => ({
      updateSession: (id: string, update: (state: ClientSessionState) => ClientSessionState) => {
        const next = update(actual.$sessionStates.get()[id]!)
        actual.$sessionStates.set({ ...actual.$sessionStates.get(), [id]: next })

        return next
      }
    })
  }
})

beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

function resetStores() {
  $activeSessionId.set(null)
  $selectedStoredSessionId.set(null)
  $sessionStates.set({})
  setCurrentModel('claude-sonnet-4.6')
  setCurrentProvider('anthropic')
  setCurrentReasoningEffort('low')
  setCurrentFastMode(false)
  setCurrentServiceTier('normal')
  setCurrentModelSource('default')
  $modelPresets.set({})
  $collapsedProviders.set([])
  $customModels.set([])
  $notifications.set([])
}

beforeEach(() => {
  vi.mocked(confirm).mockReset()
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => {
      throw new Error('unexpected HTTP request')
    })
  )
  resetStores()
})

afterEach(() => {
  cleanup()
  resetStores()
  vi.unstubAllGlobals()
})

function renderPicker({
  deferred = false,
  view = PRIMARY_SESSION_VIEW,
  switchModel,
  switchFast,
  variantFast = false,
  captureController
}: {
  deferred?: boolean
  view?: SessionView
  switchModel?: () => Promise<unknown>
  switchFast?: () => Promise<unknown>
  variantFast?: boolean
  captureController?: (controller: ReturnType<typeof useModelMenuController>['controller']) => void
} = {}) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } })
  const close = vi.fn()

  const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
    if (method === 'model.options') {
      return {
        providers: [
          { slug: 'anthropic', name: 'Anthropic', models: ['claude-sonnet-4.6'] },
          {
            slug: 'google',
            name: 'Google',
            models: variantFast ? ['gemini-3.1-pro', 'gemini-3.1-pro-fast'] : ['gemini-3.1-pro'],
            capabilities: { 'gemini-3.1-pro': { fast: !variantFast } }
          }
        ]
      }
    }

    if (method === 'config.set') {
      if (params?.key === 'model' && switchModel) {
        return switchModel()
      }

      if (params?.key === 'fast' && switchFast) {
        return switchFast()
      }

      return {
        ok: true,
        ...(deferred && params?.key !== 'fast' ? { deferred: true, status: 'deferred' } : {}),
        value: params?.value
      }
    }

    throw new Error(`unexpected gateway method: ${method}`)
  })

  function OtherController(props: ModelMenuHostProps) {
    const { controller } = useModelMenuController(props)
    captureController?.(controller)

    return null
  }

  function Harness({ controllerKey }: { controllerKey: number }) {
    const { selectModel } = useModelControls({ queryClient: client, requestGateway: requestGateway as never })

    return (
      <>
        <ModelMenuPanel onSelectModel={selectModel} requestGateway={requestGateway as never} />
        {captureController && (
          <OtherController key={controllerKey} onSelectModel={selectModel} requestGateway={requestGateway as never} />
        )}
      </>
    )
  }

  const picker = (controllerKey: number) => (
    <QueryClientProvider client={client}>
      <SessionViewProvider value={view}>
        <ModelMenuCloseContext.Provider value={close}>
          <DropdownMenu open>
            <DropdownMenuContent>
              <Harness controllerKey={controllerKey} />
            </DropdownMenuContent>
          </DropdownMenu>
        </ModelMenuCloseContext.Provider>
      </SessionViewProvider>
    </QueryClientProvider>
  )

  const mounted = render(picker(0))

  return { close, client, requestGateway, remountController: () => mounted.rerender(picker(1)) }
}

async function choose(option: 'effort' | 'Thinking' | 'Fast') {
  const trigger = (await screen.findByText(/Gemini 3\.1 Pro/i)).closest('[data-slot="dropdown-menu-sub-trigger"]')!
  fireEvent.pointerMove(trigger, { pointerType: 'mouse' })
  await screen.findByText('Effort')
  fireEvent.click(
    option === 'effort'
      ? screen.getByRole('menuitemradio', { name: 'High' })
      : screen.getByRole('switch', option === 'Fast' ? { name: 'Fast' } : { name: '' })
  )
  await act(async () => {})
}

const choices = [
  { option: 'effort' as const, effort: 'high', fast: false },
  { option: 'Thinking' as const, effort: 'none', fast: false },
  { option: 'Fast' as const, effort: 'medium', fast: true }
]

describe.each(['draft', 'idle session', 'streaming session'] as const)('%s model options', state => {
  it.each(choices)(
    '$option selects the model and applies the options for its next message',
    async ({ option, effort, fast }) => {
      const runtimeId = state === 'draft' ? null : 'runtime-owner'
      const streaming = state === 'streaming session'

      if (runtimeId) {
        $activeSessionId.set(runtimeId)
        $selectedStoredSessionId.set('stored-owner')
        $sessionStates.set({
          [runtimeId]: {
            ...createClientSessionState('stored-owner'),
            model: 'claude-sonnet-4.6',
            provider: 'anthropic',
            reasoningEffort: 'low',
            busy: streaming,
            awaitingResponse: streaming
          }
        })
      }

      const { close, client, requestGateway } = renderPicker({ deferred: streaming })
      await choose(option)

      expect(close).toHaveBeenCalledTimes(1)
      expect([$currentModel.get(), $currentProvider.get(), $currentReasoningEffort.get()]).toEqual([
        'gemini-3.1-pro',
        'google',
        effort
      ])
      expect(getModelPreset('google', 'gemini-3.1-pro')).toMatchObject({ effort, fast })
      const writes = requestGateway.mock.calls.filter(([method]) => method === 'config.set').map(([, params]) => params)

      if (runtimeId) {
        expect(writes).toEqual(
          expect.arrayContaining([
            expect.objectContaining({ session_id: runtimeId, key: 'model', value: 'gemini-3.1-pro --provider google' }),
            expect.objectContaining({ session_id: runtimeId, key: 'reasoning', value: effort }),
            expect.objectContaining({ session_id: runtimeId, key: 'fast', value: fast ? 'fast' : 'normal' })
          ])
        )
        expect(writes).toHaveLength(3)
        expect($sessionStates.get()[runtimeId]).toMatchObject({ busy: streaming, awaitingResponse: streaming })
      } else {
        expect(writes).toEqual([])
        expect(await desktopSessionCreateParams('', null)).toMatchObject({
          model: 'gemini-3.1-pro',
          provider: 'google',
          reasoning_effort: effort,
          fast
        })
      }

      expect(fetch).not.toHaveBeenCalled()
      client.clear()
    }
  )
})

it.each(choices)('$option targets the tile without changing the primary chat', async ({ option, effort, fast }) => {
  $activeSessionId.set('runtime-primary')
  $sessionStates.set({
    'runtime-primary': {
      ...createClientSessionState('stored-primary'),
      model: 'primary-model',
      provider: 'anthropic',
      reasoningEffort: 'low',
      busy: true
    },
    'runtime-tile': {
      ...createClientSessionState('stored-tile'),
      model: 'claude-sonnet-4.6',
      provider: 'anthropic',
      reasoningEffort: 'low'
    }
  })
  const empty = createClientSessionState('stored-tile')

  const field = <T,>(read: (state: ClientSessionState) => T) =>
    computed($sessionStates, states => read(states['runtime-tile'] ?? empty))

  const view: SessionView = {
    ...PRIMARY_SESSION_VIEW,
    kind: 'tile',
    $runtimeId: atom('runtime-tile'),
    $storedId: atom('stored-tile'),
    $model: field(state => state.model),
    $provider: field(state => state.provider),
    $reasoningEffort: field(state => state.reasoningEffort),
    $reasoningEffortWire: field(state => state.reasoningEffortWire ?? ''),
    $fast: field(state => state.fast),
    $serviceTier: field(state => state.serviceTier)
  }

  const { close, client, requestGateway } = renderPicker({ view })
  const primary = $sessionStates.get()['runtime-primary']
  await choose(option)
  expect(close).toHaveBeenCalledTimes(1)
  expect($sessionStates.get()['runtime-tile']).toMatchObject({
    model: 'gemini-3.1-pro',
    provider: 'google',
    reasoningEffort: effort,
    fast
  })
  expect($sessionStates.get()['runtime-primary']).toBe(primary)
  expect([$currentModel.get(), $currentProvider.get(), $currentReasoningEffort.get()]).toEqual([
    'claude-sonnet-4.6',
    'anthropic',
    'low'
  ])
  expect(requestGateway.mock.calls.filter(([method]) => method === 'config.set').map(([, params]) => params)).toEqual(
    expect.arrayContaining([
      { session_id: 'runtime-tile', key: 'model', value: 'gemini-3.1-pro --provider google --session' },
      { session_id: 'runtime-tile', key: 'reasoning', value: effort },
      { session_id: 'runtime-tile', key: 'fast', value: fast ? 'fast' : 'normal' }
    ])
  )
  client.clear()
})

it('rolls back a refused switch and leaves the picker open without applying the options', async () => {
  setCurrentModel('stale-draft-model')
  setCurrentProvider('stale-draft-provider')
  $activeSessionId.set('runtime-owner')
  $sessionStates.set({
    'runtime-owner': {
      ...createClientSessionState('stored-owner'),
      model: 'claude-sonnet-4.6',
      provider: 'anthropic',
      reasoningEffort: 'low'
    }
  })

  const { close, client, requestGateway } = renderPicker({
    switchModel: async () => {
      throw new Error('model unavailable')
    }
  })

  await choose('effort')
  expect(close).not.toHaveBeenCalled()
  expect($sessionStates.get()['runtime-owner']).toMatchObject({
    model: 'claude-sonnet-4.6',
    provider: 'anthropic',
    reasoningEffort: 'low'
  })
  expect($currentReasoningEffort.get()).toBe('low')
  expect(requestGateway.mock.calls.filter(([method]) => method === 'config.set')).toHaveLength(1)
  client.clear()
})

it('does not apply a pending selection’s options after the user changes chats', async () => {
  const reply = deferred<unknown>()
  $activeSessionId.set('runtime-owner')
  $sessionStates.set({
    'runtime-owner': {
      ...createClientSessionState('stored-owner'),
      model: 'claude-sonnet-4.6',
      provider: 'anthropic',
      reasoningEffort: 'low'
    }
  })
  const { close, client, requestGateway } = renderPicker({ switchModel: () => reply.promise })
  await choose('effort')
  await act(async () => {
    $activeSessionId.set('runtime-other')
    $selectedStoredSessionId.set('stored-other')
    $sessionStates.set({
      ...$sessionStates.get(),
      'runtime-other': {
        ...createClientSessionState('stored-other'),
        model: 'other-model',
        provider: 'other-provider',
        reasoningEffort: 'low'
      }
    })
    setCurrentModel('other-model')
    setCurrentProvider('other-provider')
    setCurrentReasoningEffort('low')
  })
  await act(async () => {
    reply.resolve({})
  })
  expect(close).not.toHaveBeenCalled()
  expect($currentReasoningEffort.get()).toBe('low')
  expect($sessionStates.get()['runtime-other']).toMatchObject({
    model: 'other-model',
    provider: 'other-provider',
    reasoningEffort: 'low'
  })
  expect(requestGateway.mock.calls.filter(([method]) => method === 'config.set')).toHaveLength(1)
  client.clear()
})

it.each([true, false])('applies the chosen options only after confirmation is accepted: %s', async accepted => {
  const answer = deferred<boolean>()
  vi.mocked(confirm).mockReturnValue(answer.promise)
  $activeSessionId.set('runtime-owner')
  $sessionStates.set({
    'runtime-owner': {
      ...createClientSessionState('stored-owner'),
      model: 'claude-sonnet-4.6',
      provider: 'anthropic',
      reasoningEffort: 'low'
    }
  })
  const switchModel = vi.fn().mockResolvedValueOnce({ confirm_required: true }).mockResolvedValueOnce({})
  const { close, client, requestGateway } = renderPicker({ switchModel })
  await choose('effort')
  expect(confirm).toHaveBeenCalledTimes(1)
  expect(close).not.toHaveBeenCalled()
  expect(requestGateway.mock.calls.filter(([method]) => method === 'config.set')).toHaveLength(1)
  await act(async () => {
    answer.resolve(accepted)
  })
  expect(close).toHaveBeenCalledTimes(accepted ? 1 : 0)
  expect($sessionStates.get()['runtime-owner']).toMatchObject({
    model: accepted ? 'gemini-3.1-pro' : 'claude-sonnet-4.6',
    reasoningEffort: accepted ? 'high' : 'low'
  })
  const writes = requestGateway.mock.calls.filter(([method]) => method === 'config.set').map(([, params]) => params)

  if (accepted) {
    expect(writes).toContainEqual({
      session_id: 'runtime-owner',
      key: 'model',
      value: 'gemini-3.1-pro --provider google',
      confirm_expensive_model: true
    })
    expect(writes).toContainEqual({ session_id: 'runtime-owner', key: 'reasoning', value: 'high' })
  } else {
    expect(writes).toHaveLength(1)
  }

  client.clear()
})

it('remembers a variant Fast choice without sending a Fast parameter to the variant model', async () => {
  const { close, client, requestGateway } = renderPicker({ variantFast: true })
  await choose('Fast')
  expect(close).toHaveBeenCalledTimes(1)
  expect($currentModel.get()).toBe('gemini-3.1-pro-fast')
  expect(getModelPreset('google', 'gemini-3.1-pro')).toMatchObject({ fast: true })
  expect(await desktopSessionCreateParams('', null)).toMatchObject({ model: 'gemini-3.1-pro-fast', fast: false })
  expect(requestGateway.mock.calls.filter(([method]) => method === 'config.set')).toHaveLength(0)
  client.clear()
})

it('surfaces a refused Fast setting and rolls it back while retaining the accepted model choice', async () => {
  $activeSessionId.set('runtime-owner')
  $sessionStates.set({
    'runtime-owner': {
      ...createClientSessionState('stored-owner'),
      model: 'claude-sonnet-4.6',
      provider: 'anthropic',
      reasoningEffort: 'low',
      busy: true
    }
  })

  const { close, client } = renderPicker({
    deferred: true,
    switchFast: async () => {
      throw new Error('fast_unavailable')
    }
  })

  await choose('Fast')
  await waitFor(() => expect($notifications.get()).toContainEqual(expect.objectContaining({ kind: 'error' })))
  expect(close).toHaveBeenCalledTimes(1)
  expect($sessionStates.get()['runtime-owner']).toMatchObject({
    model: 'gemini-3.1-pro',
    provider: 'google',
    fast: false,
    serviceTier: 'normal'
  })
  expect(getModelPreset('google', 'gemini-3.1-pro')).toMatchObject({ fast: false })
  client.clear()
})

it.each([false, true])(
  'keeps a newer option edit from another controller after the old model switch settles (remount: %s)',
  async remount => {
    const reply = deferred<unknown>()
    $activeSessionId.set('runtime-owner')
    $sessionStates.set({
      'runtime-owner': {
        ...createClientSessionState('stored-owner'),
        model: 'claude-sonnet-4.6',
        provider: 'anthropic',
        reasoningEffort: 'low'
      }
    })
    let other!: ReturnType<typeof useModelMenuController>['controller']

    const { client, requestGateway, remountController } = renderPicker({
      switchModel: () => reply.promise,
      captureController: controller => {
        other = controller
      }
    })

    await choose('effort')

    if (remount) {
      remountController()
    }

    expect(other.current.model).toBe('gemini-3.1-pro')
    await act(async () => {
      other.setOptions({ effort: 'low' }, { model: 'gemini-3.1-pro', provider: 'google', isActive: true })
    })
    await waitFor(() =>
      expect(requestGateway.mock.calls).toContainEqual([
        'config.set',
        { session_id: 'runtime-owner', key: 'reasoning', value: 'low' }
      ])
    )
    await act(async () => {
      reply.resolve({ deferred: true })
    })
    expect($sessionStates.get()['runtime-owner'].reasoningEffort).toBe('low')
    expect(getModelPreset('google', 'gemini-3.1-pro').effort).toBe('low')
    expect(requestGateway.mock.calls.filter(([, params]) => params?.key === 'reasoning')).toHaveLength(1)
    client.clear()
  }
)

it('applies an accepted deferred pick after an old-route session.info heartbeat', async () => {
  const reply = deferred<unknown>()
  $activeSessionId.set('runtime-owner')
  $sessionStates.set({
    'runtime-owner': {
      ...createClientSessionState('stored-owner'),
      model: 'claude-sonnet-4.6',
      provider: 'anthropic',
      reasoningEffort: 'low',
      busy: true
    }
  })
  const { close, client, requestGateway } = renderPicker({ switchModel: () => reply.promise })
  await choose('effort')
  await act(async () => {
    $sessionStates.set({
      ...$sessionStates.get(),
      'runtime-owner': applySessionInfoStatePatch(
        $sessionStates.get()['runtime-owner'],
        sessionInfoStatePatch({ model: 'claude-sonnet-4.6', provider: 'anthropic', reasoning_effort: 'low' })
      )
    })
    reply.resolve({ deferred: true })
  })
  expect(close).toHaveBeenCalledTimes(1)
  expect($sessionStates.get()['runtime-owner']).toMatchObject({
    model: 'gemini-3.1-pro',
    provider: 'google',
    reasoningEffort: 'high'
  })
  expect(requestGateway.mock.calls).toContainEqual([
    'config.set',
    { session_id: 'runtime-owner', key: 'reasoning', value: 'high' }
  ])
  client.clear()
})

it('preserves a matching route wire stamp when the accepted model RPC settles', async () => {
  const reply = deferred<unknown>()
  $activeSessionId.set('runtime-owner')
  $sessionStates.set({
    'runtime-owner': {
      ...createClientSessionState('stored-owner'),
      model: 'claude-sonnet-4.6',
      provider: 'anthropic'
    }
  })
  let other!: ReturnType<typeof useModelMenuController>['controller']

  const { client } = renderPicker({
    switchModel: () => reply.promise,
    captureController: controller => {
      other = controller
    }
  })

  let selection!: Promise<boolean | void>
  await act(async () => {
    selection = other.select('gemini-3.1-pro', 'google') as Promise<boolean | void>
  })
  await act(async () => {
    $sessionStates.set({
      ...$sessionStates.get(),
      'runtime-owner': applySessionInfoStatePatch(
        $sessionStates.get()['runtime-owner'],
        sessionInfoStatePatch({ model: 'gemini-3.1-pro', provider: 'google', reasoning_effort_wire: 'max' })
      )
    })
    reply.resolve({ deferred: true })
    expect(await selection).toBe(true)
  })
  expect($sessionStates.get()['runtime-owner'].reasoningEffortWire).toBe('max')
  client.clear()
})

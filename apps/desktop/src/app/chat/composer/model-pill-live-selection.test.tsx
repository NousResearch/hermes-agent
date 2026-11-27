import { useStore } from '@nanostores/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { useLayoutEffect, useRef } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import { useSessionView } from '@/app/chat/session-view'
import { useChatBarState } from '@/app/chat/use-chat-bar-state'
import { handleDeferredPickEvent } from '@/app/session/hooks/use-message-stream/gateway-event/deferred-pick'
import { handleSessionInfoEvent } from '@/app/session/hooks/use-message-stream/gateway-event/session-info'
import type { GatewayEventContext } from '@/app/session/hooks/use-message-stream/gateway-event/types'
import { useModelControls } from '@/app/session/hooks/use-model-controls'
import { useSessionStateCache } from '@/app/session/hooks/use-session-state-cache'
import { $confirmRequest, settleConfirm } from '@/store/confirm'
import * as gatewayOwner from '@/store/gateway'
import { clearNotifications } from '@/store/notifications'
import {
  $activeSessionId,
  $selectedStoredSessionId,
  clearComposerSelectionOwner,
  getComposerModelSelection,
  setAwaitingResponse,
  setBusy,
  setComposerSelectionOwner,
  setCurrentModel,
  setCurrentProvider,
  setMessages
} from '@/store/session'
import { $sessionStates, sessionTileDelegate, setSessionTileDelegate } from '@/store/session-states'
import { deferred } from '@/test/deferred'

import { ModelPill } from './model-pill'

const sid = 'existing-chat'
const provider = 'openai-codex'
const old = 'gpt-6.1-sol'
const next = 'gpt-6.2-astra'
const initialDelegate = sessionTileDelegate()

// Same path as ChatView: the live primary slice, the chat-bar snapshot and the mounted button.
// Only the owner RPC is simulated. No turn, navigation, remount or session.info is needed.
function mount(kind: 'plain' | 'tile' = 'plain') {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const reply = deferred<Record<string, unknown>>()

  const request = vi.fn((_: string, params?: Record<string, unknown>) =>
    params?.key === 'reasoning' ? Promise.resolve({ status: 'deferred' }) : reply.promise
  )

  setComposerSelectionOwner('pill-test', 'default')
  $activeSessionId.set(sid)
  setCurrentModel(old)
  setCurrentProvider(provider)

  let update!: ReturnType<typeof useSessionStateCache>['updateSessionState']

  function Surface() {
    const view = useSessionView()
    const activeSessionId = useStore(view.$runtimeId)
    const currentModel = useStore(view.$model)
    const currentProvider = useStore(view.$provider)
    const busyRef = useRef(false)

    const { updateSessionState } = useSessionStateCache({
      activeSessionId,
      busyRef,
      selectedStoredSessionId: sid,
      setAwaitingResponse,
      setBusy,
      setMessages
    })

    useLayoutEffect(() => {
      update = updateSessionState
      updateSessionState(sid, state => ({ ...state, model: old, provider }), sid)

      if (kind === 'tile') {
        updateSessionState('tile-chat', state => ({ ...state, model: old, provider, busy: true }), 'tile-chat')
      }

      setSessionTileDelegate({ updateSession: updateSessionState } as Parameters<typeof setSessionTileDelegate>[0])
    }, [updateSessionState])
    const controls = useModelControls({ queryClient: client, requestGateway: request as never })

    const pick = () => controls.selectModel({ model: next, provider, sessionId: kind === 'tile' ? 'tile-chat' : sid })

    const bar = useChatBarState({
      activeGatewayProfile: 'default',
      activeSessionId,
      contextSuggestions: [],
      currentModel,
      currentProvider,
      gateway: null,
      gatewayOpen: false,
      modelMenuContent: <span />
    })

    return (
      <>
        <button onClick={() => void pick()}>Pick target</button>
        <ModelPill disabled={false} model={bar.model} />
      </>
    )
  }

  render(
    <QueryClientProvider client={client}>
      <Surface />
    </QueryClientProvider>
  )
  const button = screen.getByRole('button', { name: /gpt-6.1-sol/i })
  expect(button.textContent).toBe('GPT-6.1-sol')
  fireEvent.click(screen.getByText('Pick target'))

  const info = (
    model: string,
    revision?: number,
    ownerProvider = provider,
    own = true,
    runtimeId = sid,
    storedId = runtimeId
  ) =>
    act(() => {
      const active = $activeSessionId.get()
      handleSessionInfoEvent({
        deps: {
          activeGatewayProfile: 'default',
          activeSessionIdRef: { current: active },
          lastCwdInfoSessionRef: { current: null },
          queryClient: client,
          scheduleSessionsRefresh: vi.fn(),
          sessionStateByRuntimeIdRef: { current: new Map(Object.entries($sessionStates.get())) },
          updateSessionState: update
        },
        event: {
          connectionId: own ? 'pill-test' : 'other-owner',
          profile: 'default',
          type: 'session.info',
          session_id: runtimeId
        },
        explicitSid: runtimeId,
        fromActiveSource: () => own,
        isActiveEvent: runtimeId === active,
        payload: { model, provider: ownerProvider, revision, stored_session_id: storedId },
        scheduleConfigRefresh: vi.fn(),
        sessionId: runtimeId
      } as unknown as GatewayEventContext)
    })

  const event = (fields: Record<string, unknown>, own = true) =>
    act(() => {
      const payload = { operation: 'model', session_id: sid, ...fields }
      handleDeferredPickEvent({
        deps: {
          activeGatewayProfile: 'default',
          activeSessionIdRef: { current: $activeSessionId.get() },
          queryClient: client,
          updateSessionState: update
        },
        event: { type: 'session.updated' },
        explicitSid: sid,
        fromActiveSource: () => own,
        payload,
        sessionId: sid
      } as unknown as GatewayEventContext)
    })

  return { button, event, info, reply, request }
}

afterEach(() => {
  cleanup()
  setSessionTileDelegate(initialDelegate as Parameters<typeof setSessionTileDelegate>[0])
  $activeSessionId.set(null)
  $selectedStoredSessionId.set(null)
  $sessionStates.set({})
  settleConfirm(false)
  clearNotifications()
  clearComposerSelectionOwner()
})

it('withdraws the queued alias when its resolved model needs confirmation and the user declines', async () => {
  using resend = vi.spyOn(gatewayOwner, 'requestGatewayForAgent').mockResolvedValue({})
  const { button, event, reply } = mount()
  await act(async () => {
    reply.resolve({ deferred: true, model: next, revision: 4 })
  })
  expect(button.textContent).toContain('next message')
  event({
    status: 'confirmation_required',
    confirm: 'test-fixture',
    target_model: 'canonical-target',
    target_provider: 'resolved-provider',
    payload: { model: next, provider },
    committed_model: old,
    committed_provider: provider
  })
  expect($confirmRequest.get()).not.toBeNull()
  expect.soft(button.textContent).toBe('GPT-6.1-sol')
  await act(async () => {
    settleConfirm(false)
  })
  expect(button.textContent).toBe('GPT-6.1-sol')
  expect($sessionStates.get()[sid].queuedModel).toBeUndefined()
  expect(getComposerModelSelection().source).toBe('default')
  expect(resend).not.toHaveBeenCalled()
})

it.each(['new connection', 'new pick on the same connection'])(
  'preserves the route for a %s if an already-sent confirmation later fails',
  async change => {
    const retry = deferred<Record<string, unknown>>()
    const resend = vi.spyOn(gatewayOwner, 'requestGatewayForAgent').mockReturnValue(retry.promise)

    try {
      const { button, event, info, reply } = mount()
      await act(async () => {
        reply.resolve({ deferred: true, model: next, revision: 4 })
      })
      event({
        status: 'confirmation_required',
        confirm: 'test-fixture',
        target_model: next,
        payload: { model: next, provider },
        committed_model: old,
        committed_provider: provider
      })
      await act(async () => {
        settleConfirm(true)
      })
      expect(resend).toHaveBeenCalledTimes(1)

      if (change === 'new connection') {
        act(() => {
          setComposerSelectionOwner('other-owner', 'default')
          setCurrentModel(next)
          setCurrentProvider('second-provider')
        })
        info(next, 6, 'second-provider')
      } else {
        event({ status: 'applied', model: next, provider: 'second-provider', revision: 6 })
      }

      expect(button.textContent).toBe('GPT-6.2-astra')
      await act(async () => {
        retry.reject(new Error('fixture rejected'))
      })
      expect(button.textContent).toBe('GPT-6.2-astra')
      expect($sessionStates.get()[sid]).toMatchObject({ model: next, provider: 'second-provider', modelRevision: 6 })
    } finally {
      resend.mockRestore()
    }
  }
)

it('changes the existing primary composer button on the applied receipt without sending a message', async () => {
  const { button, reply, request } = mount()
  await act(async () => {
    reply.resolve({ model: next, operation: 'model', provider, revision: 5, session_id: sid })
  })
  expect(request).toHaveBeenCalledTimes(1)
  expect(button.isConnected).toBe(true)
  expect(button.textContent).toBe('GPT-6.2-astra')
  expect($sessionStates.get()[sid].messages).toHaveLength(0)
})

it('names a deferred pick on the same button through old heartbeats until the owner commits it', async () => {
  const { button, event, info, reply } = mount()
  await act(async () => {
    reply.resolve({ deferred: true, model: next, revision: 4, status: 'deferred' })
  })
  const queued = 'GPT-6.2-astra applies from your next message'
  expect(button.textContent).toBe(queued)
  expect($sessionStates.get()[sid].model).toBe(old)
  info(old, 4)
  expect(button.textContent).toBe(queued)
  event({ model: next, provider, revision: 5, execution_generation: 3 })
  expect(button.textContent).toBe('GPT-6.2-astra')
  expect($sessionStates.get()[sid].model).toBe(next)
})

it('keeps a newer committed route when a pre-switch snapshot arrives late', async () => {
  const { button, info, reply } = mount()
  await act(async () => {
    reply.resolve({ model: next, provider, revision: 5 })
  })
  info(old, 4)
  expect(button.textContent).toBe('GPT-6.2-astra')
  info('gpt-6.1-luna', 6)
  expect(button.textContent).toBe('GPT-6.1-luna')
})

it('ignores an unversioned old route after a versioned commit without blocking a newer snapshot', async () => {
  const { button, info, reply } = mount()
  await act(async () => {
    reply.resolve({ model: next, provider, revision: 5 })
  })
  info(old)
  expect(button.textContent).toBe('GPT-6.2-astra')
  info(old, 5)
  expect(button.textContent).toBe('GPT-6.2-astra')
  info('gpt-6.1-luna', 6)
  expect(button.textContent).toBe('GPT-6.1-luna')
})

it('uses the confirmed next-turn receipt without waiting for a separate broadcast', async () => {
  const resend = vi
    .spyOn(gatewayOwner, 'requestGatewayForAgent')
    .mockResolvedValue({ model: next, provider, revision: 5 })

  try {
    const { button, event, reply } = mount()
    await act(async () => {
      reply.resolve({ deferred: true, revision: 4 })
    })
    event({
      status: 'confirmation_required',
      confirm: 'test-fixture',
      target_model: next,
      payload: { model: next, provider },
      committed_model: old,
      committed_provider: provider
    })
    await act(async () => {
      settleConfirm(true)
    })
    expect(resend).toHaveBeenCalledTimes(1)
    expect(button.textContent).toBe('GPT-6.2-astra')
  } finally {
    resend.mockRestore()
  }
})

it('clears a deferred label on a matching session.info even when the applied broadcast is absent', async () => {
  const { button, info, reply } = mount()
  await act(async () => {
    reply.resolve({ deferred: true, revision: 4 })
  })
  info(next, 5)
  expect(button.textContent).toBe('GPT-6.2-astra')
})

it('only the owning session can release a refused next-turn label', async () => {
  const { button, event, reply } = mount()
  await act(async () => {
    reply.resolve({ deferred: true, revision: 4 })
  })
  const refusal = { status: 'refused', model: old, provider, target_model: next }
  event(refusal, false)
  expect(button.textContent).toContain('next message')
  event(refusal)
  expect(button.textContent).toBe('GPT-6.1-sol')
})

it('displays the routed model in the receipt even when the menu selected an alias', async () => {
  const { button, reply } = mount()
  await act(async () => {
    reply.resolve({ model: 'gpt-6.2-canonical', provider, revision: 5 })
  })
  expect(button.textContent).toBe('GPT-6.2-canonical')
})

it('never reports an explicit refusal as an applied model', async () => {
  const { button, reply } = mount()
  await act(async () => {
    reply.resolve({ status: 'refused', reason: 'model_resolution_failed' })
  })
  expect(button.textContent).toBe('GPT-6.1-sol')
})

it('a next-turn tile choice leaves the running route and the primary button alone', async () => {
  const { button, reply } = mount('tile')
  await act(async () => {
    reply.resolve({ deferred: true, model: next, revision: 4 })
  })
  expect($sessionStates.get()['tile-chat']).toMatchObject({ model: old, queuedModel: { model: next, provider } })
  expect(button.textContent).toBe('GPT-6.1-sol')
})

it('updates the original live slice when its acknowledgement arrives after navigating away', async () => {
  const { button, reply } = mount()
  act(() => {
    $activeSessionId.set('other-chat')
    setCurrentModel('other-model')
  })
  await act(async () => {
    reply.resolve({ model: next, provider, revision: 5 })
  })
  expect(button.textContent).toBe('Other Model')
  act(() => {
    $activeSessionId.set(sid)
  })
  expect(button.textContent).toBe('GPT-6.2-astra')
})

it('keeps the newer committed route when a delayed refusal names that same model', async () => {
  const { button, event, reply } = mount()
  await act(async () => {
    reply.resolve({ deferred: true, model: next, revision: 4 })
  })
  event({ status: 'applied', model: next, provider, revision: 6 })
  expect(button.textContent).toBe('GPT-6.2-astra')
  event({ status: 'refused', target_model: next, model: old, provider })
  expect(button.textContent).toBe('GPT-6.2-astra')
  expect(getComposerModelSelection().source).toBe('manual')
})

it('declining an earlier owner confirmation cannot rewind a different owner with the same session id', async () => {
  const { button, event, info, reply } = mount()
  await act(async () => {
    reply.resolve({ deferred: true, revision: 4 })
  })
  event({
    status: 'confirmation_required',
    confirm: 'test-fixture',
    target_model: next,
    payload: { model: next, provider },
    committed_model: old,
    committed_provider: provider
  })
  act(() => {
    setComposerSelectionOwner('other-owner', 'default')
    setCurrentModel(next)
    setCurrentProvider(provider)
  })
  info(next, 6)
  expect(button.textContent).toBe('GPT-6.2-astra')
  await act(async () => {
    settleConfirm(false)
  })
  expect(button.textContent).toBe('GPT-6.2-astra')
})

it('does not confirm an old dialog on another owner with the same session id and displayed route', async () => {
  const resend = vi
    .spyOn(gatewayOwner, 'requestGatewayForAgent')
    .mockResolvedValue({ model: next, provider, revision: 5 })

  try {
    const { event, reply } = mount()
    await act(async () => {
      reply.resolve({ deferred: true, revision: 4 })
    })
    event({
      status: 'confirmation_required',
      confirm: 'test-fixture',
      target_model: next,
      payload: { model: next, provider },
      committed_model: old,
      committed_provider: provider
    })
    act(() => {
      setComposerSelectionOwner('other-owner', 'default')
      setCurrentModel(old)
      setCurrentProvider(provider)
    })
    await act(async () => {
      settleConfirm(true)
    })
    expect(resend).not.toHaveBeenCalled()
  } finally {
    resend.mockRestore()
  }
})

it('ignores a different owner’s higher-revision snapshot for this id but still updates its other background session', async () => {
  const { button, info, reply } = mount()
  await act(async () => {
    reply.resolve({ model: next, provider, revision: 5 })
  })
  info(old, 50, provider, false)
  expect(button.textContent).toBe('GPT-6.2-astra')
  info('background', 51, provider, false, 'other-chat')
  expect($sessionStates.get()['other-chat'].model).toBe('background')
  info(old, 6)
  expect(button.textContent).toBe('GPT-6.1-sol')
})

it('keeps the open runtime when a different owner announces a rebuild of the same stored conversation', async () => {
  const { button, info, reply } = mount()
  await act(async () => {
    reply.resolve({ model: next, provider, revision: 5 })
  })
  act(() => {
    $selectedStoredSessionId.set(sid)
  })
  info(old, 51, provider, false, 'their-runtime', sid)
  expect($activeSessionId.get()).toBe(sid)
  expect(button.textContent).toBe('GPT-6.2-astra')
})

it('shows the committed route while a previously deferred choice awaits confirmation', async () => {
  const { button, event, reply } = mount()
  await act(async () => {
    reply.resolve({ deferred: true, revision: 4 })
  })
  event({
    status: 'confirmation_required',
    confirm: 'test-fixture',
    target_model: next,
    payload: { model: next, provider },
    committed_model: old,
    committed_provider: provider
  })
  expect($confirmRequest.get()).not.toBeNull()
  expect(button.textContent).toBe('GPT-6.1-sol')
  await act(async () => {
    settleConfirm(false)
  })
  expect(button.textContent).toBe('GPT-6.1-sol')
})

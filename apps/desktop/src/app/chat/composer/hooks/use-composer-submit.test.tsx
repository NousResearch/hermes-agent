import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, renderHook, waitFor } from '@testing-library/react'
import { type RefObject, useCallback, useEffect, useRef, useState } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { useMessageStream } from '@/app/session/hooks/use-message-stream'
import { usePromptActions } from '@/app/session/hooks/use-prompt-actions'
import type { ClientSessionState } from '@/app/types'
import type { ComposerAttachment } from '@/store/composer'
import { createClientSessionState } from '@/lib/chat-runtime'
import { clearQueuedPrompts, getQueuedPrompts } from '@/store/composer-queue'
import { setAwaitingResponse, setBusy } from '@/store/session'
import type { RpcEvent, TurnOrigin } from '@/types/hermes'

import type { QueueEditState } from '../composer-utils'

import { useComposerQueue } from './use-composer-queue'
import { useComposerSubmit } from './use-composer-submit'

vi.mock('@/hermes', () => ({
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  PROMPT_SUBMIT_REQUEST_TIMEOUT_MS: 1_800_000,
  setApiRequestProfile: vi.fn(),
  transcribeAudio: vi.fn()
}))

interface SubmitHarnessOptions {
  attachments?: ComposerAttachment[]
  busy?: boolean
  compacting?: boolean
  text?: string
}

function renderSubmitHook({
  attachments = [],
  busy = false,
  compacting = false,
  text = ''
}: SubmitHarnessOptions = {}) {
  const draftRef = { current: text }
  const editor = document.createElement('div')
  editor.dataset.slot = 'composer-rich-input'
  editor.textContent = text
  const editorRef = { current: editor }
  const onCancel = vi.fn()
  const onSteer = vi.fn(async () => true)
  const onSubmit = vi.fn(async () => true)
  const queueCurrentDraft = vi.fn(() => true)

  const clearDraft = vi.fn(() => {
    draftRef.current = ''
    editorRef.current!.textContent = ''
  })

  const hook = renderHook(() =>
    useComposerSubmit({
      activeQueueSessionKey: 'stored-session',
      activeQueueSessionKeyRef: { current: 'stored-session' },
      attachments,
      busy,
      compacting,
      clearDraft,
      disabled: false,
      draftRef,
      drainNextQueued: vi.fn(async () => false),
      editorRef,
      exitQueuedEdit: vi.fn(() => false),
      focusInput: vi.fn(),
      inputDisabled: false,
      loadIntoComposer: vi.fn(),
      onCancel,
      onSteer,
      onSubmit,
      queueCurrentDraft,
      queueEdit: null,
      queuedPrompts: [],
      sessionId: 'runtime-session',
      setComposerText: vi.fn(),
      stashAt: vi.fn(),
      turnOrigin: null
    })
  )

  return { clearDraft, hook, onCancel, onSteer, onSubmit, queueCurrentDraft }
}

describe('useComposerSubmit busy-turn routing', () => {
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('steers a plain-text follow-up instead of queueing or stopping', async () => {
    const { hook, onCancel, onSteer, onSubmit, queueCurrentDraft } = renderSubmitHook({
      busy: true,
      text: 'change course'
    })

    act(() => {
      hook.result.current.submitDraft()
    })

    await waitFor(() => expect(onSteer).toHaveBeenCalledWith('change course'))
    expect(queueCurrentDraft).not.toHaveBeenCalled()
    expect(onCancel).not.toHaveBeenCalled()
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('queues a plain-text follow-up while the active turn is compacting', () => {
    const { hook, onCancel, onSteer, onSubmit, queueCurrentDraft } = renderSubmitHook({
      busy: true,
      compacting: true,
      text: 'wait for the summary'
    })

    act(() => {
      hook.result.current.submitDraft()
    })

    expect(queueCurrentDraft).toHaveBeenCalledTimes(1)
    expect(onSteer).not.toHaveBeenCalled()
    expect(onSubmit).not.toHaveBeenCalled()
    expect(onCancel).not.toHaveBeenCalled()
  })

  it('runs slash commands immediately while busy', async () => {
    const { clearDraft, hook, onCancel, onSteer, onSubmit, queueCurrentDraft } = renderSubmitHook({
      busy: true,
      text: '/compress preserve context'
    })

    act(() => {
      hook.result.current.submitDraft()
    })

    await waitFor(() => expect(onSubmit).toHaveBeenCalledWith('/compress preserve context'))
    expect(clearDraft).toHaveBeenCalledTimes(1)
    expect(onSteer).not.toHaveBeenCalled()
    expect(queueCurrentDraft).not.toHaveBeenCalled()
    expect(onCancel).not.toHaveBeenCalled()
  })

  it('queues an attachment-bearing follow-up while busy', () => {
    const attachment: ComposerAttachment = { id: 'doc', kind: 'file', label: 'notes.txt' }

    const { hook, onCancel, onSteer, onSubmit, queueCurrentDraft } = renderSubmitHook({
      attachments: [attachment],
      busy: true,
      text: 'read this'
    })

    act(() => {
      hook.result.current.submitDraft()
    })

    expect(queueCurrentDraft).toHaveBeenCalledTimes(1)
    expect(onSteer).not.toHaveBeenCalled()
    expect(onSubmit).not.toHaveBeenCalled()
    expect(onCancel).not.toHaveBeenCalled()
  })

  it('stops an active turn only with an empty composer', () => {
    const { hook, onCancel, onSteer, onSubmit, queueCurrentDraft } = renderSubmitHook({ busy: true })

    act(() => {
      hook.result.current.submitDraft()
    })

    expect(onCancel).toHaveBeenCalledTimes(1)
    expect(onSteer).not.toHaveBeenCalled()
    expect(onSubmit).not.toHaveBeenCalled()
    expect(queueCurrentDraft).not.toHaveBeenCalled()
  })

  it('submits a normal turn while idle', async () => {
    const { hook, onCancel, onSteer, onSubmit, queueCurrentDraft } = renderSubmitHook({ text: 'ordinary question' })

    act(() => {
      hook.result.current.submitDraft()
    })

    await waitFor(() => expect(onSubmit).toHaveBeenCalledWith('ordinary question', { attachments: [] }))
    expect(onSteer).not.toHaveBeenCalled()
    expect(queueCurrentDraft).not.toHaveBeenCalled()
    expect(onCancel).not.toHaveBeenCalled()
  })
})

function renderSubmit(turnOrigin: TurnOrigin) {
  const queueCurrentDraft = vi.fn(() => true)
  const onCancel = vi.fn(async () => undefined)
  const onSubmit = vi.fn(async () => true)
  const editorRef: RefObject<HTMLDivElement | null> = { current: null }
  const draftRef: RefObject<string> = { current: 'Human question' }

  const hook = renderHook(() =>
    useComposerSubmit({
      activeQueueSessionKey: 'stored-session',
      activeQueueSessionKeyRef: { current: 'stored-session' },
      attachments: [],
      busy: true,
      compacting: false,
      clearDraft: vi.fn(),
      disabled: false,
      draftRef,
      drainNextQueued: vi.fn(async () => false),
      editorRef,
      exitQueuedEdit: vi.fn(() => false),
      focusInput: vi.fn(),
      inputDisabled: false,
      loadIntoComposer: vi.fn(),
      onCancel,
      onSteer: undefined,
      onSubmit,
      queueCurrentDraft,
      queueEdit: null,
      queuedPrompts: [],
      sessionId: 'runtime-session',
      setComposerText: vi.fn(),
      stashAt: vi.fn(),
      turnOrigin
    })
  )

  return { hook, onCancel, onSubmit, queueCurrentDraft }
}

describe('notification-origin foreground priority', () => {
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('queues a normal draft and immediately interrupts a notification turn', () => {
    const { hook, onCancel, onSubmit, queueCurrentDraft } = renderSubmit('notification')

    act(() => hook.result.current.submitDraft())

    expect(queueCurrentDraft).toHaveBeenCalledOnce()
    expect(onCancel).toHaveBeenCalledOnce()
    expect(onCancel).toHaveBeenCalledWith({ preserveBusyUntilSettled: true })
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it.each<TurnOrigin>(['user', 'goal'])('keeps %s turns on the existing queue-only path', turnOrigin => {
    const { hook, onCancel, onSubmit, queueCurrentDraft } = renderSubmit(turnOrigin)

    act(() => hook.result.current.submitDraft())

    expect(queueCurrentDraft).toHaveBeenCalledOnce()
    expect(onCancel).not.toHaveBeenCalled()
    expect(onSubmit).not.toHaveBeenCalled()
  })
})

const RUNTIME_SESSION_ID = 'runtime-notification-session'
const STORED_SESSION_ID = 'stored-notification-session'

interface PreemptionHarnessHandle {
  getState: () => ClientSessionState
  handleGatewayEvent: (event: RpcEvent) => void
  submitDraft: () => void
}

function NotificationPreemptionHarness({
  onReady,
  requestGateway
}: {
  onReady: (handle: PreemptionHarnessHandle) => void
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
}) {
  const initialStateRef = useRef<ClientSessionState>({
    ...createClientSessionState(STORED_SESSION_ID),
    awaitingResponse: true,
    busy: true,
    interrupted: false,
    turnOrigin: 'notification'
  })

  const stateRef = useRef(initialStateRef.current)
  const [state, setState] = useState(initialStateRef.current)
  const activeSessionIdRef = useRef<string | null>(RUNTIME_SESSION_ID)
  const selectedStoredSessionIdRef = useRef<string | null>(STORED_SESSION_ID)
  const busyRef = useRef(true)
  const sessionStateByRuntimeIdRef = useRef(new Map([[RUNTIME_SESSION_ID, initialStateRef.current]]))
  const queryClientRef = useRef(new QueryClient())
  const draftRef = useRef('Human question')
  const editorRef = useRef<HTMLDivElement | null>(null)
  const queueEditRef = useRef<QueueEditState | null>(null)

  const updateSessionState = useCallback(
    (_sessionId: string, updater: (current: ClientSessionState) => ClientSessionState) => {
      const next = updater(stateRef.current)
      stateRef.current = next
      sessionStateByRuntimeIdRef.current.set(RUNTIME_SESSION_ID, next)
      busyRef.current = next.busy
      setBusy(next.busy)
      setAwaitingResponse(next.awaitingResponse)
      setState(next)

      return next
    },
    []
  )

  const actions = usePromptActions({
    activeSessionId: RUNTIME_SESSION_ID,
    activeSessionIdRef,
    branchCurrentSession: async () => true,
    busyRef,
    createBackendSessionForSend: async () => RUNTIME_SESSION_ID,
    getRouteToken: () => 'notification-preemption',
    handleSkinCommand: () => '',
    openMemoryGraph: () => undefined,
    refreshSessions: async () => undefined,
    requestGateway,
    resumeStoredSession: () => undefined,
    selectedStoredSessionIdRef,
    startFreshSessionDraft: () => undefined,
    sttEnabled: false,
    updateSessionState
  })

  const queue = useComposerQueue({
    activeQueueSessionKey: STORED_SESSION_ID,
    attachments: [],
    busy: state.busy,
    clearDraft: () => {
      draftRef.current = ''
    },
    draftRef,
    focusInput: () => undefined,
    loadIntoComposer: () => undefined,
    onCancel: actions.cancelRun,
    onSubmit: actions.submitText,
    queueEditRef,
    queueSessionKey: STORED_SESSION_ID,
    sessionId: RUNTIME_SESSION_ID
  })

  const submit = useComposerSubmit({
    activeQueueSessionKey: STORED_SESSION_ID,
    activeQueueSessionKeyRef: { current: STORED_SESSION_ID },
    attachments: [],
    busy: state.busy,
    compacting: false,
    clearDraft: () => {
      draftRef.current = ''
    },
    disabled: false,
    draftRef,
    drainNextQueued: queue.drainNextQueued,
    editorRef,
    exitQueuedEdit: queue.exitQueuedEdit,
    focusInput: () => undefined,
    inputDisabled: false,
    loadIntoComposer: () => undefined,
    onCancel: actions.cancelRun,
    onSteer: undefined,
    onSubmit: actions.submitText,
    queueCurrentDraft: queue.queueCurrentDraft,
    queueEdit: queue.queueEdit,
    queuedPrompts: queue.queuedPrompts,
    sessionId: RUNTIME_SESSION_ID,
    setComposerText: () => undefined,
    stashAt: () => undefined,
    turnOrigin: state.turnOrigin ?? null
  })

  const stream = useMessageStream({
    activeSessionIdRef,
    hydrateFromStoredSession: async () => undefined,
    queryClient: queryClientRef.current,
    refreshHermesConfig: async () => undefined,
    refreshSessions: async () => undefined,
    sessionStateByRuntimeIdRef,
    updateSessionState
  })

  useEffect(() => {
    onReady({
      getState: () => stateRef.current,
      handleGatewayEvent: stream.handleGatewayEvent,
      submitDraft: submit.submitDraft
    })
  }, [onReady, stream.handleGatewayEvent, submit.submitDraft])

  return null
}

describe('notification preemption integration', () => {
  afterEach(() => {
    cleanup()
    clearQueuedPrompts(STORED_SESSION_ID)
    setBusy(false)
    setAwaitingResponse(false)
    vi.restoreAllMocks()
  })

  it('drains one queued user turn only after the interrupted notification completes', async () => {
    const calls: Array<{ method: string; params?: Record<string, unknown> }> = []

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      return {} as never
    })

    let handle: PreemptionHarnessHandle | null = null

    const onReady = (next: PreemptionHarnessHandle) => {
      handle = next
    }

    setBusy(true)
    setAwaitingResponse(true)
    render(<NotificationPreemptionHarness onReady={onReady} requestGateway={requestGateway} />)
    await waitFor(() => expect(handle).not.toBeNull())

    act(() => handle!.submitDraft())

    await waitFor(() => expect(calls.filter(call => call.method === 'session.interrupt')).toHaveLength(1))
    expect(handle!.getState()).toMatchObject({ busy: true, interrupted: true })
    expect(calls.filter(call => call.method === 'prompt.submit')).toHaveLength(0)
    expect(getQueuedPrompts(STORED_SESSION_ID)).toHaveLength(1)

    act(() =>
      handle!.handleGatewayEvent({
        payload: { status: 'interrupted', text: '', turn_origin: 'notification' },
        session_id: RUNTIME_SESSION_ID,
        type: 'message.complete'
      })
    )

    await waitFor(() => expect(calls.filter(call => call.method === 'prompt.submit')).toHaveLength(1))
    expect(calls.filter(call => call.method === 'prompt.submit')[0]?.params).toMatchObject({
      session_id: RUNTIME_SESSION_ID,
      text: 'Human question'
    })
    await waitFor(() => expect(getQueuedPrompts(STORED_SESSION_ID)).toHaveLength(0))
    expect(calls.filter(call => call.method === 'prompt.submit')).toHaveLength(1)
    expect(handle!.getState().turnOrigin).toBe('user')
  })

  it('releases the preemption gate when the notification interrupt RPC fails', async () => {
    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.interrupt') {
        throw new Error('interrupt unavailable')
      }

      return {} as never
    })

    let handle: PreemptionHarnessHandle | null = null

    const onReady = (next: PreemptionHarnessHandle) => {
      handle = next
    }

    setBusy(true)
    setAwaitingResponse(true)
    render(<NotificationPreemptionHarness onReady={onReady} requestGateway={requestGateway} />)
    await waitFor(() => expect(handle).not.toBeNull())

    act(() => handle!.submitDraft())

    await waitFor(() =>
      expect(requestGateway).toHaveBeenCalledWith('session.interrupt', { session_id: RUNTIME_SESSION_ID })
    )
    await waitFor(() =>
      expect(requestGateway.mock.calls.filter(([method]) => method === 'prompt.submit')).toHaveLength(1)
    )
    expect(getQueuedPrompts(STORED_SESSION_ID)).toHaveLength(0)
  })
})

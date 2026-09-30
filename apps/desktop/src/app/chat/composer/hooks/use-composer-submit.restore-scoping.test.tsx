import { useStore } from '@nanostores/react'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { startTransition, Suspense, useLayoutEffect, useRef } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import { createSessionRpcDispatcher } from '@/app/contrib/session-rpc-dispatcher'
import { NEW_CHAT_ROUTE, routeSessionId, sessionRoute } from '@/app/routes'
import { usePromptActions } from '@/app/session/hooks/use-prompt-actions'
import { useSessionActions } from '@/app/session/hooks/use-session-actions'
import { useSessionStateCache } from '@/app/session/hooks/use-session-state-cache'
import {
  announceNewSessionDraftKey,
  clearSessionDraft,
  mainComposerScope,
  stashSessionDraft,
  takeSessionDraft
} from '@/store/composer'
import { requestGatewayForAgent, requestGatewayForProfile } from '@/store/gateway'
import { $newChatRoute } from '@/store/profile'
import {
  $activeSessionId,
  $selectedStoredSessionId,
  _resetSessionOwnerHintsForTests,
  setActiveSessionId,
  setAwaitingResponse,
  setBusy,
  setMessages,
  setSelectedStoredSessionId,
  setSessions,
  setYoloActive
} from '@/store/session'
import { _resetSessionOwnerHoldsForTests } from '@/store/session-states'

import { composerPlainText } from '../rich-editor'
import type { ChatBarProps } from '../types'

import { useComposerDraft } from './use-composer-draft'
import { useComposerSubmit } from './use-composer-submit'

// Substitute the assistant-ui adapter and transport, not the draft DOM,
// subscription, persistence, creation, submit actions, cache or dispatcher.
// One test injects onSubmit to exercise a direct rejected promise too.
const composer = vi.hoisted(() => {
  let text = ''
  const listeners = new Set<() => void>()

  const runtime = {
    getState: () => ({ text }),
    subscribe: (listener: () => void) => {
      listeners.add(listener)

      return () => {
        listeners.delete(listener)
      }
    }
  }

  const api = {
    setText: (value: string) => {
      text = value
      listeners.forEach(listener => listener())
    }
  }

  return { aui: { composer: () => api }, runtime }
})

vi.mock('@assistant-ui/react', () => ({
  useAui: () => composer.aui,
  useAuiState: (selector: (state: { composer: { text: string } }) => unknown) =>
    selector({ composer: composer.runtime.getState() }),
  useComposerRuntime: () => composer.runtime
}))
vi.mock('@/store/gateway', async original => ({
  ...(await original<Record<string, unknown>>()),
  requestGatewayForAgent: vi.fn(),
  requestGatewayForProfile: vi.fn(),
  retainGatewayForAgent: vi.fn(async () => () => {}),
  retainGatewayForSessionTurn: vi.fn(async () => () => {})
}))
vi.mock('@/store/profile', async original => ({
  ...(await original<Record<string, unknown>>()),
  ensureGatewayAgent: vi.fn(async () => undefined),
  ensureGatewayProfile: vi.fn(async () => undefined)
}))

let handles: {
  cache: ReturnType<typeof useSessionStateCache>
  draft: ReturnType<typeof useComposerDraft>
  submit: ReturnType<typeof useComposerSubmit>
}

let route: string | null = 'stored-B'
const busyRef = { current: false }
const getRouteToken = () => `${route ? sessionRoute(route) : NEW_CHAT_ROUTE}::`

interface HarnessProps {
  pendingScope?: string
  suspend?: Promise<void>
  onSubmit?: ChatBarProps['onSubmit']
}

function PendingNavigation({ suspend }: Pick<HarnessProps, 'suspend'>) {
  if (suspend) {
    throw suspend
  }

  return null
}

function Harness({ pendingScope, suspend, onSubmit }: HarnessProps) {
  const active = useStore($activeSessionId)
  const selected = useStore($selectedStoredSessionId)
  const attachments = useStore(mainComposerScope.$attachments)
  const queueEditRef = useRef(null)
  const creatingSessionRef = useRef(false)

  const cache = useSessionStateCache({
    activeSessionId: active,
    selectedStoredSessionId: selected,
    busyRef,
    setBusy,
    setAwaitingResponse,
    setMessages
  })

  const request = createSessionRpcDispatcher({
    ambientRequest: async () => {
      throw new Error('unexpected ambient request')
    },
    runtimeIdByStoredSessionIdRef: cache.runtimeIdByStoredSessionIdRef,
    selectedStoredSessionIdRef: cache.selectedStoredSessionIdRef,
    sessionStateByRuntimeIdRef: cache.sessionStateByRuntimeIdRef
  })

  const sessionActions = useSessionActions({
    activeSessionId: active,
    selectedStoredSessionId: selected,
    ...cache,
    busyRef,
    creatingSessionRef,
    getRoutedStoredSessionId: () => route,
    getRouteToken,
    navigate: to => {
      route = routeSessionId(String(to))
    },
    requestGateway: request
  })

  const actions = usePromptActions({
    activeSessionId: active,
    ...cache,
    busyRef,
    branchCurrentSession: async () => false,
    createBackendSessionForSend: sessionActions.createBackendSessionForSend,
    getRoutedStoredSessionId: () => route,
    getRouteToken,
    handleSkinCommand: () => '',
    openMemoryGraph: () => {},
    refreshSessions: async () => {},
    requestGateway: request,
    resumeStoredSession: async () => {
      throw new Error('unexpected resume')
    },
    startFreshSessionDraft: () => {},
    sttEnabled: false
  })

  const activeQueueSessionKey = pendingScope ?? selected

  const draft = useComposerDraft({
    activeQueueSessionKey,
    focusKey: active,
    inputDisabled: false,
    queueEditRef,
    sessionId: active
  })

  const submit = useComposerSubmit({
    ...draft,
    activeQueueSessionKey,
    attachments,
    busy: false,
    compacting: false,
    disabled: false,
    inputDisabled: false,
    drainNextQueued: async () => false,
    exitQueuedEdit: () => false,
    onCancel: () => {},
    onSteer: actions.redirectPrompt,
    onSteerHidden: actions.injectHiddenPrompt,
    onSubmit: onSubmit ?? actions.submitText,
    queueCurrentDraft: () => false,
    queueEdit: null,
    queuedPrompts: [],
    sessionId: active
  })

  // Keep the test's actions bound to the committed tree during a transition.
  useLayoutEffect(() => {
    handles = { cache, draft, submit }
  })

  return (
    <>
      <div contentEditable data-testid="editor" ref={draft.editorRef} />
      <PendingNavigation suspend={suspend} />
    </>
  )
}

function view(props: HarnessProps = {}) {
  return (
    <Suspense fallback={<div>Pending navigation</div>}>
      <Harness {...props} />
    </Suspense>
  )
}

function navigate(session: 'A' | 'B' | null) {
  route = session ? `stored-${session}` : null
  setSelectedStoredSessionId(route)
  setActiveSessionId(session ? `rt-${session}` : null)
  busyRef.current = false
  setBusy(false)
}

function seed(props: HarnessProps = {}, initial: 'B' | null = 'B') {
  setSessions(
    (initial ? ['A', 'B'] : ['A']).map(id => ({
      id: `stored-${id}`,
      profile: 'default',
      source: 'desktop',
      message_count: 1,
      ended_at: null,
      input_tokens: 0,
      output_tokens: 0,
      is_active: true,
      last_active: 1,
      started_at: 1,
      model: null,
      preview: null,
      title: null,
      tool_call_count: 0
    }))
  )
  navigate(initial)
  const rendered = render(view(props))
  act(() => {
    handles.cache.ensureSessionState('rt-A', 'stored-A')

    if (initial) {
      handles.cache.ensureSessionState('rt-B', 'stored-B')
    }
  })

  return rendered
}

const editorText = () => composerPlainText(handles.draft.editorRef.current!).trim()
const flushDraft = () => act(() => window.dispatchEvent(new Event('pagehide')))

afterEach(() => {
  cleanup()
  mainComposerScope.clear()
  clearSessionDraft('stored-A')
  clearSessionDraft('stored-B')
  clearSessionDraft(null)
  _resetSessionOwnerHintsForTests({ storage: true })
  _resetSessionOwnerHoldsForTests()
  setYoloActive(false)
  announceNewSessionDraftKey(null)
  $newChatRoute.set(null)
  window.localStorage.clear()
  vi.resetAllMocks()
  setActiveSessionId(null)
  setSelectedStoredSessionId(null)
  setSessions([])
  setBusy(false)
  setAwaitingResponse(false)
  setMessages([])
})

it('keeps a late rejected B submit out of A’s draft and next ordinary Send (#66661)', async () => {
  let rejectSubmit!: (error: Error) => void
  vi.mocked(requestGatewayForProfile).mockImplementation(async (_profile, method) => {
    if (method !== 'prompt.submit') {
      throw new Error(`unexpected ${method}`)
    }

    return await new Promise((_resolve, reject) => {
      rejectSubmit = reject
    })
  })
  stashSessionDraft('stored-A', 'A’s own next question', [])
  stashSessionDraft('stored-B', 'B’s research report', [])
  seed()
  act(() => handles.submit.submitDraft())
  await waitFor(() =>
    expect(requestGatewayForProfile).toHaveBeenCalledWith(
      'default',
      'prompt.submit',
      expect.objectContaining({ session_id: 'rt-B', text: 'B’s research report' }),
      expect.anything(),
      undefined
    )
  )

  act(() => navigate('A'))
  expect(editorText()).toBe('A’s own next question')
  // The real action converts a transport rejection into accepted === false.
  await act(async () => {
    rejectSubmit(new Error('fixture transport rejected B submit'))
  })
  expect(editorText()).toBe('A’s own next question')
  flushDraft()
  expect(takeSessionDraft('stored-A').text.trim()).toBe('A’s own next question')
  expect(takeSessionDraft('stored-B').text.trim()).toBe('B’s research report')

  vi.mocked(requestGatewayForProfile).mockResolvedValue({ status: 'streaming' })
  act(() => handles.submit.submitDraft())
  await waitFor(() => expect(requestGatewayForProfile).toHaveBeenCalledTimes(2))
  expect(
    vi.mocked(requestGatewayForProfile).mock.calls.map(([, method, params]) => ({
      method,
      session: params?.session_id,
      text: params?.text
    }))
  ).toEqual([
    { method: 'prompt.submit', session: 'rt-B', text: 'B’s research report' },
    { method: 'prompt.submit', session: 'rt-A', text: 'A’s own next question' }
  ])
  act(() => navigate('B'))
  expect(editorText()).toBe('B’s research report')
})

it('restores text and attachments by the loaded draft owner, not an uncommitted navigation render', async () => {
  let rejectSubmit!: (error: Error) => void

  const onSubmit = vi.fn<ChatBarProps['onSubmit']>(
    () =>
      new Promise<boolean>((_resolve, reject) => {
        rejectSubmit = reject
      })
  )

  const attachmentA = { id: 'url-a', kind: 'url' as const, label: 'A reference' }
  const attachmentB = { id: 'url-b', kind: 'url' as const, label: 'B reference' }
  stashSessionDraft('stored-A', 'draft A', [attachmentA])
  stashSessionDraft('stored-B', 'draft B', [attachmentB])
  const rendered = seed({ onSubmit })
  act(() => handles.submit.submitDraft())
  expect(onSubmit).toHaveBeenCalledWith(expect.stringContaining('draft B'), {
    attachments: [attachmentB],
    composerScope: 'stored-B'
  })
  act(() => navigate('A'))

  // React starts rendering B again but cannot commit it. The render-time
  // queue ref now says B; the actual editor and draft-swap owner remain A.
  const suspended = new Promise<void>(() => {})
  await act(async () => {
    startTransition(() => rendered.rerender(view({ onSubmit, pendingScope: 'stored-B', suspend: suspended })))
  })
  expect(handles.draft.activeQueueSessionKeyRef.current).toBe('stored-B')
  expect(editorText()).toBe('draft A')
  await act(async () => {
    rejectSubmit(new Error('fixture rejected onSubmit promise'))
  })
  expect(editorText()).toBe('draft A')
  expect(mainComposerScope.$attachments.get()).toEqual([attachmentA])
  flushDraft()
  expect(takeSessionDraft('stored-A').text.trim()).toBe('draft A')
  expect(takeSessionDraft('stored-A').attachments).toEqual([attachmentA])
  expect(takeSessionDraft('stored-B').text.trim()).toBe('draft B')
  expect(takeSessionDraft('stored-B').attachments).toEqual([attachmentB])

  // Abandon the pending render, then really switch. Ordinary draft restore
  // recovers B; a rejection while B stays loaded must still repaint it.
  rendered.rerender(view({ onSubmit }))
  act(() => navigate('B'))
  expect(editorText()).toBe('draft B')
  expect(mainComposerScope.$attachments.get()).toEqual([attachmentB])
  await act(async () => {
    startTransition(() => rendered.rerender(view({ onSubmit, pendingScope: 'stored-A', suspend: suspended })))
  })
  expect(handles.draft.activeQueueSessionKeyRef.current).toBe('stored-A')
  onSubmit.mockResolvedValueOnce(false)
  await act(async () => handles.submit.submitDraft())
  expect(onSubmit).toHaveBeenLastCalledWith(expect.stringContaining('draft B'), {
    attachments: [attachmentB],
    composerScope: 'stored-B'
  })
  expect(editorText()).toBe('draft B')
  expect(mainComposerScope.$attachments.get()).toEqual([attachmentB])
  expect(takeSessionDraft('stored-B').text.trim()).toBe('draft B')
})

it.each([
  { destination: 'B', beforeCreateReturns: false },
  { destination: 'A', beforeCreateReturns: false },
  { destination: null, beforeCreateReturns: false },
  { destination: 'A', beforeCreateReturns: true }
] as const)(
  'restores a rejected real first send to its created scope ($destination, before create returns: $beforeCreateReturns)',
  async ({ destination, beforeCreateReturns }) => {
    const attachment = { id: 'url-first', kind: 'url' as const, label: 'first reference' }
    stashSessionDraft(null, 'my first question', [attachment])
    stashSessionDraft('stored-A', 'A’s own draft', [])
    $newChatRoute.set({ connectionId: 'connection-first-send', profile: 'default' })
    setYoloActive(beforeCreateReturns)

    let rejectRequest!: (error: Error) => void
    vi.mocked(requestGatewayForAgent).mockImplementation(async (_connection, _profile, method) => {
      if (method === 'session.create') {
        return { session_id: 'rt-B', stored_session_id: 'stored-B', info: {} }
      }

      if (method === (beforeCreateReturns ? 'config.set' : 'prompt.submit')) {
        return new Promise((_resolve, reject) => {
          rejectRequest = reject
        })
      }

      throw new Error(`unexpected ${method}`)
    })

    seed({}, null)
    act(() => handles.submit.submitDraft())
    await waitFor(() => expect(rejectRequest).toBeTypeOf('function'))
    expect(route).toBe('stored-B')
    expect(handles.draft.draftScopeRef.current).toBe('stored-B')
    expect(editorText()).toBe('')
    expect(mainComposerScope.$attachments.get()).toEqual([])
    expect(vi.mocked(requestGatewayForAgent).mock.calls.map(([, , method]) => method)).toEqual([
      'session.create',
      beforeCreateReturns ? 'config.set' : 'prompt.submit'
    ])

    if (destination !== 'B') {
      // A different stored chat OR a new pre-session draft is not the owner
      // of this submit, even though this submit originally captured null.
      stashSessionDraft(null, 'another fresh draft', [])
      act(() => navigate(destination))
    }

    // Creation can await an armed-YOLO config write after it assigns the
    // stored key. Its eventual drift abort must still restore to that key.
    await act(async () => rejectRequest(new Error('fixture transport rejected first send')))
    expect(editorText()).toBe(
      destination === 'B' ? 'my first question' : destination === 'A' ? 'A’s own draft' : 'another fresh draft'
    )
    expect(mainComposerScope.$attachments.get()).toEqual(destination === 'B' ? [attachment] : [])
    flushDraft()
    expect(takeSessionDraft('stored-B').text.trim()).toBe('my first question')
    expect(takeSessionDraft('stored-B').attachments).toEqual([attachment])
    expect(takeSessionDraft('stored-A').text.trim()).toBe('A’s own draft')
    expect(takeSessionDraft(null).text.trim()).toBe(destination === 'B' ? '' : 'another fresh draft')
    act(() => navigate('B'))
    expect(editorText()).toBe('my first question')
    expect(mainComposerScope.$attachments.get()).toEqual([attachment])
  }
)

it('does not treat arbitrary navigation during a real create as assignment of the rejected draft', async () => {
  stashSessionDraft(null, 'unsent first question', [])
  stashSessionDraft('stored-A', 'A’s own draft', [])
  $newChatRoute.set({ connectionId: 'connection-aborted-create', profile: 'default' })
  let completeCreate!: (value: unknown) => void
  vi.mocked(requestGatewayForAgent).mockImplementation(async (_connection, _profile, method) => {
    if (method === 'session.create') {
      return new Promise(resolve => {
        completeCreate = resolve
      })
    }

    if (method === 'session.close') {
      return {}
    }

    throw new Error(`unexpected ${method}`)
  })

  seed({}, null)
  act(() => handles.submit.submitDraft())
  await waitFor(() => expect(completeCreate).toBeTypeOf('function'))
  act(() => navigate('A'))
  await act(async () => completeCreate({ session_id: 'rt-B', stored_session_id: 'stored-B', info: {} }))
  expect(vi.mocked(requestGatewayForAgent).mock.calls.map(([, , method]) => method)).toEqual([
    'session.create',
    'session.close'
  ])
  expect(route).toBe('stored-A')
  expect(editorText()).toBe('A’s own draft')
  flushDraft()
  expect(takeSessionDraft('stored-A').text.trim()).toBe('A’s own draft')
  expect(takeSessionDraft('stored-B').text.trim()).toBe('')
  expect(takeSessionDraft(null).text.trim()).toBe('unsent first question')
  act(() => navigate(null))
  expect(editorText()).toBe('unsent first question')
})

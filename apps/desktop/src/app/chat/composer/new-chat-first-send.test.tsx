import { useStore } from '@nanostores/react'
import { act, cleanup, render } from '@testing-library/react'
import { useRef } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import {
  announceNewSessionDraftKey,
  clearSessionDraft,
  type ComposerAttachment,
  mainComposerScope,
  rotateFreshDraftKey
} from '@/store/composer'
import { $queuedPromptsBySession, enqueueQueuedPrompt, getQueuedPrompts } from '@/store/composer-queue'
import { setSessionsLoading } from '@/store/session'
import { $sessionStates } from '@/store/session-states'

import type { QueueEditState } from './composer-utils'
import { useComposerDraft } from './hooks/use-composer-draft'
import { useComposerQueue } from './hooks/use-composer-queue'
import { useComposerQueueScopeKey } from './queue-slot'
import type { ChatBarProps } from './types'

vi.mock('@assistant-ui/react', () => ({
  useAui: () => ({ composer: () => ({ setText: () => undefined }) }),
  useAuiState: (selector: (state: { composer: { text: string } }) => unknown) => selector({ composer: { text: '' } }),
  useComposerRuntime: () => ({ getState: () => ({ text: '' }), subscribe: () => () => undefined })
}))

afterEach(() => {
  cleanup()
  $queuedPromptsBySession.set({})
  $sessionStates.set({})
  mainComposerScope.clear()
  clearSessionDraft('stored-new')
  announceNewSessionDraftKey(null)
  localStorage.clear()
})

interface Scope {
  busy: boolean
  queueSessionKey: null | string
  sessionId: null | string
}

type Composer = ReturnType<typeof useComposerDraft> & ReturnType<typeof useComposerQueue> & { scope: null | string }

// ChatBar's draft + queue spine on the scope it actually resolves.
function Harness({ freshDraftKey, latest, onSubmit, ...scope }: Scope & {
  freshDraftKey: string
  latest: { current: Composer | null }
  onSubmit: ChatBarProps['onSubmit']
}) {
  const activeQueueSessionKey = useComposerQueueScopeKey(scope.queueSessionKey, scope.sessionId, freshDraftKey)
  const queueEditRef = useRef<QueueEditState | null>(null)
  const attachments = useStore(mainComposerScope.$attachments)

  const draft = useComposerDraft({
    activeQueueSessionKey,
    focusKey: null,
    inputDisabled: false,
    queueEditRef,
    sessionId: scope.sessionId
  })

  const queue = useComposerQueue({
    activeQueueSessionKey,
    attachments,
    busy: scope.busy,
    clearDraft: draft.clearDraft,
    draftRef: draft.draftRef,
    focusInput: draft.focusInput,
    loadIntoComposer: draft.loadIntoComposer,
    onCancel: vi.fn(),
    onSteer: undefined,
    onSubmit,
    queueEditRef,
    queueSessionKey: scope.queueSessionKey,
    sessionId: scope.sessionId
  })

  latest.current = { ...draft, ...queue, scope: activeQueueSessionKey }

  return null
}

it('a new chat first send keeps its creation-time draft, queues on the new chat and leaves session A alone (#139190)', async () => {
  setSessionsLoading(false)
  const freshDraftKey = rotateFreshDraftKey()
  const onSubmit = vi.fn<ChatBarProps['onSubmit']>(async () => true)
  const latest: { current: Composer | null } = { current: null }
  const file: ComposerAttachment = { id: 'file:notes', kind: 'file', label: 'notes.txt' }

  $sessionStates.set({ 'session-a': { storedSessionId: 'session-a' } as ClientSessionState })
  enqueueQueuedPrompt('session-a', { attachments: [], id: 'for-a', text: 'for A' })

  const view = render(
    <Harness busy freshDraftKey={freshDraftKey} latest={latest} onSubmit={onSubmit} queueSessionKey="session-a" sessionId="session-a" />
  )

  const show = (scope: Scope) =>
    act(() => view.rerender(<Harness freshDraftKey={freshDraftKey} latest={latest} onSubmit={onSubmit} {...scope} />))

  // New chat: the route has no stored key yet while A's runtime is still active for a frame.
  show({ busy: true, queueSessionKey: null, sessionId: 'session-a' })
  expect(latest.current?.scope).toBe(freshDraftKey)
  expect(latest.current?.queuedPrompts).toEqual([])

  // First send is in flight (session.create); the user keeps typing and attaches a file.
  act(() => {
    latest.current!.draftRef.current = 'and also this'
    mainComposerScope.$attachments.set([file])
  })

  // Creation assigns the stored id: announce, bind the runtime, route onto it.
  announceNewSessionDraftKey('stored-new')
  $sessionStates.set({
    'session-a': { storedSessionId: 'session-a' } as ClientSessionState,
    'rt-new': { storedSessionId: 'stored-new' } as ClientSessionState
  })
  show({ busy: true, queueSessionKey: 'stored-new', sessionId: 'rt-new' })

  expect(latest.current?.scope).toBe('stored-new')
  expect(latest.current?.draftRef.current).toBe('and also this')
  expect(mainComposerScope.$attachments.get()).toEqual([file])

  // The first turn is running: a follow-up queues on the new chat.
  act(() => {
    latest.current!.draftRef.current = 'follow-up'
    mainComposerScope.$attachments.set([])
  })
  await act(async () => {
    await latest.current!.queueCurrentDraft()
  })
  expect(getQueuedPrompts('stored-new').map(entry => entry.text)).toEqual(['follow-up'])

  // The turn settles: the follow-up drains into the new chat, nothing of A moves or sends.
  show({ busy: false, queueSessionKey: 'stored-new', sessionId: 'rt-new' })
  await vi.waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1))
  expect(onSubmit.mock.calls[0]?.[0]).toBe('follow-up')
  expect(onSubmit.mock.calls[0]?.[1]).toMatchObject({ sessionId: 'rt-new', storedSessionId: 'stored-new' })
  expect(getQueuedPrompts('session-a').map(entry => entry.id)).toEqual(['for-a'])
  expect(getQueuedPrompts(freshDraftKey)).toEqual([])
})

it('a follow-up queued while session.create is in flight moves to the new chat and sends there', async () => {
  setSessionsLoading(false)
  const freshDraftKey = rotateFreshDraftKey()
  const onSubmit = vi.fn<ChatBarProps['onSubmit']>(async () => true)
  const latest: { current: Composer | null } = { current: null }

  $sessionStates.set({ 'session-a': { storedSessionId: 'session-a' } as ClientSessionState })
  enqueueQueuedPrompt('session-a', { attachments: [], id: 'for-a', text: 'for A' })

  const view = render(
    <Harness busy freshDraftKey={freshDraftKey} latest={latest} onSubmit={onSubmit} queueSessionKey={null} sessionId="session-a" />
  )

  const show = (scope: Scope) =>
    act(() => view.rerender(<Harness freshDraftKey={freshDraftKey} latest={latest} onSubmit={onSubmit} {...scope} />))

  expect(latest.current?.scope).toBe(freshDraftKey)

  // session.create is in flight: the follow-up can only land under the chat's fresh draft key.
  act(() => {
    latest.current!.draftRef.current = 'follow-up'
  })
  await act(async () => {
    await latest.current!.queueCurrentDraft()
  })
  expect(getQueuedPrompts(freshDraftKey).map(entry => entry.text)).toEqual(['follow-up'])

  // Creation assigns the stored id: announce, bind the runtime, route onto it.
  announceNewSessionDraftKey('stored-new')
  $sessionStates.set({
    'session-a': { storedSessionId: 'session-a' } as ClientSessionState,
    'rt-new': { storedSessionId: 'stored-new' } as ClientSessionState
  })
  show({ busy: true, queueSessionKey: 'stored-new', sessionId: 'rt-new' })

  expect(latest.current?.scope).toBe('stored-new')
  expect(getQueuedPrompts('stored-new').map(entry => entry.text)).toEqual(['follow-up'])
  expect(getQueuedPrompts(freshDraftKey)).toEqual([])

  // The first turn settles: the follow-up drains into the new chat; A is untouched.
  show({ busy: false, queueSessionKey: 'stored-new', sessionId: 'rt-new' })
  await vi.waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1))
  expect(onSubmit.mock.calls[0]?.[0]).toBe('follow-up')
  expect(onSubmit.mock.calls[0]?.[1]).toMatchObject({ sessionId: 'rt-new', storedSessionId: 'stored-new' })
  expect(getQueuedPrompts('session-a').map(entry => entry.id)).toEqual(['for-a'])
  expect(getQueuedPrompts(freshDraftKey)).toEqual([])
})

it('claiming the runtime re-scopes the composer without a sessionId or route change', () => {
  const freshDraftKey = rotateFreshDraftKey()
  const latest: { current: Composer | null } = { current: null }
  const onSubmit = vi.fn<ChatBarProps['onSubmit']>(async () => true)

  // session.create published the runtime before the mirror: nothing claims it yet.
  render(
    <Harness busy freshDraftKey={freshDraftKey} latest={latest} onSubmit={onSubmit} queueSessionKey={null} sessionId="rt-new" />
  )
  expect(latest.current?.scope).toBe('rt-new')

  // The mirror lands; props and route stay put.
  act(() => $sessionStates.set({ 'rt-new': { storedSessionId: 'stored-new' } as ClientSessionState }))

  expect(latest.current?.scope).toBe(freshDraftKey)
})

it('a session already claimed elsewhere never hands its queue to the new chat on creation', () => {
  const freshDraftKey = rotateFreshDraftKey()
  const latest: { current: Composer | null } = { current: null }
  const onSubmit = vi.fn<ChatBarProps['onSubmit']>(async () => true)

  $sessionStates.set({ 'session-a': { storedSessionId: 'session-a' } as ClientSessionState })
  enqueueQueuedPrompt('session-a', { attachments: [], id: 'for-a', text: 'for A' })

  const view = render(
    <Harness busy freshDraftKey={freshDraftKey} latest={latest} onSubmit={onSubmit} queueSessionKey="session-a" sessionId="session-a" />
  )

  // A stale announcement naming another chat while A is routed: A's queue stays put.
  announceNewSessionDraftKey('stored-new')
  act(() =>
    view.rerender(
      <Harness busy freshDraftKey={freshDraftKey} latest={latest} onSubmit={onSubmit} queueSessionKey="stored-new" sessionId="rt-new" />
    )
  )

  expect(getQueuedPrompts('session-a').map(entry => entry.id)).toEqual(['for-a'])
  expect(getQueuedPrompts('stored-new')).toEqual([])
})

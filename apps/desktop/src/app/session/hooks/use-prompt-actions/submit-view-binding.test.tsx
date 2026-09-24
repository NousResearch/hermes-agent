import { useStore } from '@nanostores/react'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { PRIMARY_SESSION_VIEW } from '@/app/chat/session-view'
import { type ChatMessage, chatMessageText } from '@/lib/chat-messages'
import {
  $activeSessionId,
  $messages,
  setActiveSessionId,
  setActiveSessionStoredIdRotation,
  setSelectedStoredSessionId,
  setSessions
} from '@/store/session'
import { clearAllSessionStates } from '@/store/session-states'
import type { SessionInfo } from '@/types/hermes'

import { useSessionStateCache } from '../use-session-state-cache'

import { clearSingleFlightSessionResumeState } from './single-flight-resume'

import { usePromptActions } from '.'

vi.mock('@/hermes', () => ({
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  getSession: vi.fn(),
  PROMPT_SUBMIT_REQUEST_TIMEOUT_MS: 1_800_000,
  setApiRequestProfile: vi.fn(),
  transcribeAudio: vi.fn()
}))

vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestGatewayForAgent: vi.fn()
}))

const STORED = 'stored-b'
const STALE_RUNTIME = 'rt-stale'
const RESUMED_RUNTIME = 'rt-resumed'

const row = (id: string, role: ChatMessage['role'], text: string): ChatMessage => ({
  id,
  role,
  parts: [{ type: 'text', text }]
})

const EARLIER = [row('u1', 'user', 'earlier prompt'), row('a1', 'assistant', 'earlier reply')]

type Cache = ReturnType<typeof useSessionStateCache>
type Submit = (text: string) => Promise<boolean>

// Real session-state cache + real prompt actions, with the pane's active id
// read from the same atom PRIMARY_SESSION_VIEW reads. Assertions are on what
// the chat RENDERS, not on the legacy `$messages` mirror.
function Harness({
  onReady,
  requestGateway,
  selectedStoredSessionIdRef
}: {
  onReady: (submit: Submit, cache: Cache) => void
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
  selectedStoredSessionIdRef: MutableRefObject<string | null>
}) {
  const activeSessionId = useStore($activeSessionId)
  const busyRef: MutableRefObject<boolean> = { current: false }

  const cache = useSessionStateCache({
    activeSessionId,
    busyRef,
    selectedStoredSessionId: selectedStoredSessionIdRef.current,
    setAwaitingResponse: () => undefined,
    setBusy: () => undefined,
    setMessages: messages => $messages.set(messages)
  })

  const actions = usePromptActions({
    activeSessionId,
    activeSessionIdRef: cache.activeSessionIdRef,
    branchCurrentSession: async () => true,
    busyRef,
    createBackendSessionForSend: async () => null,
    getRoutedStoredSessionId: () => null,
    getRuntimeIdForStoredSession: cache.getRuntimeIdForStoredSession,
    getRouteToken: () => 'token',
    handleSkinCommand: () => '',
    openMemoryGraph: () => undefined,
    refreshSessions: async () => undefined,
    requestGateway,
    resumeStoredSession: () => undefined,
    runtimeIdByStoredSessionIdRef: cache.runtimeIdByStoredSessionIdRef,
    selectedStoredSessionIdRef,
    startFreshSessionDraft: () => undefined,
    sttEnabled: false,
    updateSessionState: cache.updateSessionState
  })

  onReady(text => act(async () => actions.submitText(text)) as Promise<boolean>, cache)

  return null
}

async function mountOnStaleRuntime() {
  const selectedStoredSessionIdRef: MutableRefObject<string | null> = { current: STORED }
  setSelectedStoredSessionId(STORED)
  setActiveSessionId(STALE_RUNTIME)

  const requestGateway = vi.fn(
    async (method: string) => (method === 'session.resume' ? { session_id: RESUMED_RUNTIME } : {}) as never
  )

  let submit!: Submit
  let cache!: Cache

  render(
    <Harness
      onReady={(s, c) => {
        submit = s
        cache = c
      }}
      requestGateway={requestGateway}
      selectedStoredSessionIdRef={selectedStoredSessionIdRef}
    />
  )
  await waitFor(() => expect(submit).toBeDefined())

  return { cache, requestGateway, submit: (text: string) => submit(text) }
}

const rendered = () => PRIMARY_SESSION_VIEW.$messages.get().map(chatMessageText)

describe('a submit that resumes the selected session moves the chat to the resumed runtime (#71733, #117867)', () => {
  beforeEach(() => {
    clearSingleFlightSessionResumeState()
    clearAllSessionStates()
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    clearAllSessionStates()
    $messages.set([])
    setActiveSessionId(null)
    setActiveSessionStoredIdRotation(null)
    setSelectedStoredSessionId(null)
    setSessions([])
  })

  it('paints the prompt and keeps the thread when the ownership proof was lost', async () => {
    const { cache, requestGateway, submit } = await mountOnStaleRuntime()

    act(() => {
      cache.updateSessionState(STALE_RUNTIME, state => ({ ...state, messages: EARLIER }), STORED)
    })
    // The stored→runtime entry is gone (evicted, reconnect, rotation), so
    // submit's entry-time ownership check refuses the pane's runtime and
    // resumes the stored session, which answers with a NEW runtime id.
    cache.runtimeIdByStoredSessionIdRef.current.delete(STORED)

    expect(rendered()).toEqual(['earlier prompt', 'earlier reply'])

    await submit('the prompt that vanished')

    expect(requestGateway).toHaveBeenCalledWith(
      'prompt.submit',
      expect.objectContaining({ session_id: RESUMED_RUNTIME }),
      expect.anything()
    )
    expect($activeSessionId.get()).toBe(RESUMED_RUNTIME)
    expect(rendered()).toEqual(['earlier prompt', 'earlier reply', 'the prompt that vanished'])
  })

  it('follows the resumed runtime after a compression rotation the selection never followed', async () => {
    setSessions(() => [{ _lineage_root_id: STORED, id: `${STORED}-next` } as SessionInfo])

    const { cache, submit } = await mountOnStaleRuntime()

    act(() => {
      cache.updateSessionState(STALE_RUNTIME, state => ({ ...state, messages: EARLIER }), STORED)
      // Auto-compression rotates the stored id on the live runtime. The
      // reverse entry for the old id is dropped; the pane selection still
      // names the old id.
      cache.updateSessionState(STALE_RUNTIME, state => state, `${STORED}-next`)
    })

    await submit('asked again after compaction')

    expect($activeSessionId.get()).toBe(RESUMED_RUNTIME)
    expect(rendered()).toEqual(['earlier prompt', 'earlier reply', 'asked again after compaction'])
  })

  it('never carries another conversation into the resumed runtime', async () => {
    const { cache, submit } = await mountOnStaleRuntime()

    act(() => {
      // The pane's runtime is proven to belong to a DIFFERENT stored session.
      cache.updateSessionState(STALE_RUNTIME, state => ({ ...state, messages: EARLIER }), 'stored-other')
    })

    await submit('first prompt in B')

    expect($activeSessionId.get()).toBe(RESUMED_RUNTIME)
    expect(rendered()).toEqual(['first prompt in B'])
  })
})

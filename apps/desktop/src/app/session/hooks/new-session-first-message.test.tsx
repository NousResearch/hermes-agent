import { useStore } from '@nanostores/react'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'
import type * as GatewayStore from '@/store/gateway'
import { requestGatewayForProfile } from '@/store/gateway'
import {
  $activeSessionId,
  $messages,
  $selectedStoredSessionId,
  setActiveSessionId,
  setAwaitingResponse,
  setBusy,
  setMessages,
  setSelectedStoredSessionId,
  setSessions
} from '@/store/session'
import { installRestBridge } from '@/test/rest-bridge'

import { useSessionActions } from './use-session-actions'
import { useSessionStateCache } from './use-session-state-cache'

// The row is default-profile owned, so its session RPCs ride the profile socket; route those to the fake.
vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<typeof GatewayStore>()),
  requestGatewayForProfile: vi.fn()
}))

const PROMPT = 'Plan my week.\n\nAbout me:\n- Call me Sid.'

type Options = Parameters<typeof useSessionActions>[0]

// The test resumes the routed id itself, the step use-route-resume takes after the navigate.
const stayOnRoute: Options['navigate'] = () => {}

interface Ready {
  messagesOf: (runtimeId: string) => readonly ChatMessage[]
  resume: ReturnType<typeof useSessionActions>['resumeSession']
  submitNew: ReturnType<typeof useSessionActions>['submitTextToNewSession']
}

// Real session cache + real view sync: the user row has to survive the route's warm resume.
function Harness({
  onReady,
  requestGateway
}: {
  onReady: (ready: Ready) => void
  requestGateway: Options['requestGateway']
}) {
  const activeSessionId = useStore($activeSessionId)
  const selectedStoredSessionId = useStore($selectedStoredSessionId)
  const busyRef = useRef(false)
  const creatingSessionRef = useRef(false)

  const cache = useSessionStateCache({
    activeSessionId,
    busyRef,
    selectedStoredSessionId,
    setAwaitingResponse,
    setBusy,
    setMessages
  })

  const actions = useSessionActions({
    activeSessionId,
    activeSessionIdRef: cache.activeSessionIdRef,
    busyRef,
    creatingSessionRef,
    ensureSessionState: cache.ensureSessionState,
    getRouteToken: () => 'new-session',
    getRoutedStoredSessionId: () => null,
    holdSessionTranscriptView: cache.holdSessionTranscriptView,
    navigate: stayOnRoute,
    requestGateway,
    routedSessionId: null,
    resetViewSync: cache.resetViewSync,
    runtimeIdByStoredSessionIdRef: cache.runtimeIdByStoredSessionIdRef,
    selectedStoredSessionId,
    selectedStoredSessionIdRef: cache.selectedStoredSessionIdRef,
    sessionStateByRuntimeIdRef: cache.sessionStateByRuntimeIdRef,
    syncSessionStateToView: cache.syncSessionStateToView,
    updateSessionState: cache.updateSessionState
  })

  useEffect(() => {
    onReady({
      messagesOf: runtimeId => cache.sessionStateByRuntimeIdRef.current.get(runtimeId)?.messages ?? [],
      resume: actions.resumeSession,
      submitNew: actions.submitTextToNewSession
    })
  }, [actions.resumeSession, actions.submitTextToNewSession, cache, onReady])

  return null
}

const userRows = (messages: readonly ChatMessage[]) => messages.filter(message => message.role === 'user')

const ACTIVATED = {
  info: {},
  message_count: 0,
  messages: [],
  messages_omitted: true,
  resumed: 'stored-new',
  running: true,
  session_id: 'rt-new',
  session_key: 'stored-new'
}

// GET /api/sessions/stored-new/messages: the first user row is not readable there yet when the route lands.
let persisted: object[] = []

async function mount({ refuse = false } = {}) {
  const wire = (method: string): object => {
    if (method === 'prompt.submit' && refuse) {
      throw new Error('refused')
    }

    return method === 'session.create'
      ? { session_id: 'rt-new', stored_session_id: 'stored-new' }
      : method === 'session.activate'
        ? ACTIVATED
        : {}
  }

  const calls: Parameters<Options['requestGateway']>[] = []

  const requestGateway: Options['requestGateway'] = async <T,>(...call: Parameters<Options['requestGateway']>) => {
    calls.push(call)
    const answer = wire(call[0])

    // SAFETY: each answer above is the wire shape of the method that asked for it.
    return answer as T
  }

  const rest = installRestBridge(request =>
    request.path.endsWith('/messages') ? { messages: persisted, session_id: 'stored-new' } : {}
  )

  let ready!: Ready
  render(<Harness onReady={value => (ready = value)} requestGateway={requestGateway} />)
  await waitFor(() => expect(ready).toBeDefined())
  vi.mocked(requestGatewayForProfile).mockImplementation((_profile, method, params) => requestGateway(method, params))

  return { calls, ready, rest }
}

describe('submitTextToNewSession first message', () => {
  beforeEach(() => {
    persisted = []
  })

  afterEach(() => {
    cleanup()
    setActiveSessionId(null)
    setSelectedStoredSessionId(null)
    setMessages([])
    setSessions([])
    vi.restoreAllMocks()
  })

  it('shows the submitted text as the first user message once the route opens the new chat', async () => {
    const { calls, ready } = await mount()

    await act(async () => {
      await ready.submitNew(PROMPT)
    })

    // The navigate to #/stored-new lands here: use-route-resume resumes the routed id.
    await act(async () => {
      await ready.resume('stored-new', true)
    })

    await waitFor(() => expect(userRows($messages.get())).toHaveLength(1))
    expect($activeSessionId.get()).toBe('rt-new')
    expect(userRows($messages.get())[0]?.parts).toEqual([expect.objectContaining({ text: PROMPT, type: 'text' })])
    expect(calls).toContainEqual(['prompt.submit', { session_id: 'rt-new', text: PROMPT }])
  })

  it('keeps one first message when the persisted transcript already has it', async () => {
    persisted = [{ content: PROMPT, role: 'user', timestamp: 1 }]
    const { ready, rest } = await mount()

    await act(async () => {
      await ready.submitNew(PROMPT)
    })

    await act(async () => {
      await ready.resume('stored-new', true)
    })

    await waitFor(() =>
      expect(rest).toHaveBeenCalledWith(expect.objectContaining({ path: expect.stringContaining('/messages') }))
    )
    await waitFor(() => expect(userRows($messages.get())).toHaveLength(1))
  })

  it('drops the first message when the prompt is refused', async () => {
    const { ready } = await mount({ refuse: true })

    await act(async () => {
      await expect(ready.submitNew(PROMPT)).rejects.toThrow('refused')
    })

    expect(userRows(ready.messagesOf('rt-new'))).toHaveLength(0)
  })
})

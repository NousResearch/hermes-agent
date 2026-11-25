import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import type { GatewayEvent } from '@hermes/shared'
import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { type ReactNode, useRef } from 'react'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { rehydrateLiveSessionStatuses } from '@/app/contrib/hooks/use-background-sync'
import { useMessageStream } from '@/app/session/hooks/use-message-stream'
import { useSessionStateCache } from '@/app/session/hooks/use-session-state-cache'
import { PendingApprovalStack } from '@/components/assistant-ui/tool/approval'
import type { HermesGateway } from '@/hermes'
import { $gateway } from '@/store/gateway'
import { respondToApprovalAction } from '@/store/native-notifications'
import { $profileDotStateByScope, profileDotSummaryFor } from '@/store/profile-dot-state'
import { clearAllPrompts, replayPendingApproval, sessionApprovalRequests } from '@/store/prompts'
import { resetBackgroundPollingGuard } from '@/store/runtime-gone'
import { resetServerRequestsForTests } from '@/store/server-requests'
import { setActiveSessionId, setAwaitingResponse, setBusy, setMessages, setSessions } from '@/store/session'
import { $sessionDotStateById } from '@/store/session-dot-state'
import {
  $attentionSessionIds,
  clearAllSessionStates,
  liveSessionScopes,
  recordSessionEventScope
} from '@/store/session-states'
import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'
import type { SessionInfo } from '@/types/hermes'

// A folder or site card outlives the turn that raised it. The chat it belongs to is marked as
// waiting for an answer for as long as the card is parked, and no longer: in the sidebar, on the
// profile's square, and in the set of gateways the app keeps open. These cases drive the real
// stream handlers, stores and card.

const RUNTIME = 'runtime-1'
const STORED = 'stored-1'
const noop = async () => undefined
const FOLDER = { access: 'read', choices: ['once', 'always', 'deny'], command: '', kind: 'access', path: '/srv/data' }

let stream: ReturnType<typeof useMessageStream>
let pendingList: Record<string, unknown>[] = []

function Runtime({ children }: { children: ReactNode }) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({ messages: [], isRunning: false, onNew: async () => {} })

  return <AssistantRuntimeProvider runtime={runtime}>{children}</AssistantRuntimeProvider>
}

function Chat() {
  const busyRef = useRef(false)
  const queryClient = useRef(new QueryClient()).current

  const cache = useSessionStateCache({
    activeSessionId: RUNTIME,
    selectedStoredSessionId: STORED,
    busyRef,
    setBusy,
    setAwaitingResponse,
    setMessages
  })

  stream = useMessageStream({
    ...cache,
    queryClient,
    hydrateFromStoredSession: noop,
    refreshHermesConfig: noop,
    refreshSessions: noop
  })

  return (
    <Runtime>
      <PendingApprovalStack />
    </Runtime>
  )
}

const send = (type: GatewayEvent['type'], payload: GatewayEvent['payload'] = {}) =>
  act(() => stream.handleGatewayEvent({ session_id: RUNTIME, type, payload }))

const settle = () => act(() => new Promise<void>(resolve => setTimeout(resolve, 60)))

function raise(params: Record<string, unknown>) {
  act(() => {
    stream.handleServerRequest({
      fail: vi.fn(),
      id: 'srq-1',
      method: 'approval',
      params: { request_id: 'card-1', session_id: RUNTIME, ...params },
      profile: 'default',
      respond: vi.fn()
    } as never)
  })
}

const turnStarts = () => send('message.start', {})
const folderCard = () => raise(FOLDER)
const turnCompletes = () => send('message.complete', { text: 'done' })

const toolCompletes = () => {
  send('tool.start', { name: 'read_file', tool_id: 't1', args: {} })
  send('tool.complete', { name: 'read_file', tool_id: 't1' })
}

async function askPending(approvals: Record<string, unknown>[]) {
  pendingList = approvals
  await act(() => replayPendingApproval($gateway.get() as never, RUNTIME))
  await settle()
}

/** Open the chat. `listed` puts its row in the sidebar; `owned` records which gateway serves it;
 *  `reported: false` is a chat the window has heard nothing from but an empty heartbeat. */
function openChat({
  listed = true,
  owned = true,
  reported = true
}: Partial<Record<'listed' | 'owned' | 'reported', boolean>> = {}) {
  setSessions(
    listed ? [{ id: STORED, title: 'A chat', ...(owned ? { profile: 'default' } : {}) } as unknown as SessionInfo] : []
  )
  setActiveSessionId(RUNTIME)

  if (owned) {
    recordSessionEventScope({ connectionId: 'local', profile: 'default', session_id: RUNTIME })
  }

  $gateway.set({
    request: vi.fn(async (method: string) =>
      method === 'approval.pending' ? { approvals: pendingList } : { ok: true }
    )
  } as unknown as HermesGateway)
  render(<Chat />)
  // The window knows this chat: its runtime is bound to the stored id of the sidebar row.
  send('session.info', { running: false, stored_session_id: STORED, ...(reported ? { model: 'model-1' } : {}) })
}

const sidebarDot = () => $sessionDotStateById.get()[STORED]
const inAttentionSet = () => $attentionSessionIds.get().includes(STORED)
const profileDot = () => profileDotSummaryFor($profileDotStateByScope.get(), 'local', 'default')?.state
const cards = () => sessionApprovalRequests(RUNTIME).get().length

beforeAll(() => {
  stubMenuDomApis()
  stubResizeObserver()
})

afterEach(() => {
  cleanup()
  clearAllPrompts()
  clearAllSessionStates()
  resetServerRequestsForTests()
  resetBackgroundPollingGuard()
  setSessions([])
  setActiveSessionId(null)
  $gateway.set(null)
  pendingList = []
})

describe('a chat with a folder card parked stays marked as waiting', () => {
  it.each<[string, () => void]>([
    ['a tool of the same turn completes', toolCompletes],
    ['the turn completes', turnCompletes],
    ['the turn ends with an error', () => send('error', { message: 'boom' })],
    [
      'the turn completes and the background poll lists the session as idle',
      () => {
        turnCompletes()
        act(() =>
          rehydrateLiveSessionStatuses({ sessions: [{ id: RUNTIME, session_key: STORED, status: 'idle' }] } as never)
        )
      }
    ]
  ])('after %s', async (_name, then) => {
    openChat()
    turnStarts()
    folderCard()
    then()
    await settle()

    expect(cards()).toBe(1)
    expect(sidebarDot()).toBe('needs-input')
    expect(inAttentionSet()).toBe(true)
    expect(profileDot()).toBe('needs-input')
    // The gateway the answer goes to stays among the gateways the app keeps open.
    expect(liveSessionScopes().size).toBe(1)
  })

  it.each([
    ['a chat that has reported its state', true],
    ['a chat the window has only an empty heartbeat from', false]
  ])('when the card is raised while no turn runs, for %s', async (_name, reported) => {
    openChat({ reported })
    folderCard()
    await settle()

    expect(sidebarDot()).toBe('needs-input')
    expect(profileDot()).toBe('needs-input')
  })

  it('when approval.pending restores the card', async () => {
    openChat({ owned: false })
    await askPending([{ ...FOLDER, request_id: 'card-restored' }])

    expect(cards()).toBe(1)
    expect(sidebarDot()).toBe('needs-input')
    expect(inAttentionSet()).toBe(true)
  })

  it('on the profile square, for a chat that has no row in the sidebar', async () => {
    openChat({ listed: false })
    turnStarts()
    folderCard()
    turnCompletes()
    await settle()

    expect(profileDot()).toBe('needs-input')
  })
})

describe('the mark goes when the card goes', () => {
  it.each<[string, () => Promise<void> | void]>([
    ['the person clicks Allow on the card', () => fireEvent.click(screen.getByRole('button', { name: /^Allow/ }))],
    ['the person clicks Deny on the notification', () => act(() => respondToApprovalAction(RUNTIME, 'reject:card-1'))],
    [
      'the sender withdraws the card on timeout',
      () => send('request.cancel', { id: 'srq-1', method: 'approval', reason: 'timeout' })
    ],
    [
      'the sender withdraws the card as answered elsewhere',
      () => send('request.cancel', { id: 'srq-1', method: 'approval', reason: 'resolved' })
    ]
  ])('%s, with no turn running', async (_name, cardGoes) => {
    openChat()
    folderCard()
    await settle()
    expect(sidebarDot()).toBe('needs-input')

    await cardGoes()
    await settle()
    // The next report from the backend must not bring the mark back.
    send('session.info', { running: false, stored_session_id: STORED, model: 'model-2' })

    expect(cards()).toBe(0)
    expect(sidebarDot()).not.toBe('needs-input')
    expect(inAttentionSet()).toBe(false)
    expect(profileDot()).not.toBe('needs-input')
    expect(liveSessionScopes().size).toBe(0)
  })

  it('the person clicks Allow while the turn that raised the card still runs', async () => {
    openChat()
    turnStarts()
    folderCard()
    fireEvent.click(screen.getByRole('button', { name: /^Allow/ }))
    await settle()

    expect(cards()).toBe(0)
    expect(sidebarDot()).toBe('working')
    expect(inAttentionSet()).toBe(false)
  })

  it('approval.pending no longer lists the card', async () => {
    openChat({ owned: false })
    folderCard()
    await settle()
    expect(sidebarDot()).toBe('needs-input')

    await askPending([])

    expect(cards()).toBe(0)
    expect(sidebarDot()).not.toBe('needs-input')
    expect(inAttentionSet()).toBe(false)
  })
})

describe('a command card keeps the waiting mark of its turn', () => {
  it('marks the chat while the turn waits, and the mark ends with the turn', async () => {
    openChat()
    turnStarts()
    raise({ command: 'rm -rf build', description: 'delete the build folder' })

    expect(sidebarDot()).toBe('needs-input')
    expect(profileDot()).toBe('needs-input')

    turnCompletes()
    await settle()

    expect(cards()).toBe(0)
    expect(sidebarDot()).not.toBe('needs-input')
    expect(inAttentionSet()).toBe(false)
  })
})

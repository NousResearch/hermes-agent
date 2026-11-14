import { JsonRpcGatewayError } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { act, cleanup, render } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { createSessionRpcDispatcher } from '@/app/contrib/session-rpc-dispatcher'
import { sessionRoute } from '@/app/routes'
import { textPart } from '@/lib/chat-messages'
import { requestGatewayForAgent } from '@/store/gateway'
import {
  $activeSessionId,
  $selectedStoredSessionId,
  setActiveSessionId,
  setActiveSessionStoredIdRotation,
  setAwaitingResponse,
  setBusy,
  setMessages,
  setSelectedStoredSessionId,
  setSessions
} from '@/store/session'
import { clearAllSessionStates } from '@/store/session-states'
import type { SessionInfo } from '@/types/hermes'

import { useSessionStateCache } from '../use-session-state-cache'

import { clearSingleFlightSessionResumeState } from './single-flight-resume'

import { usePromptActions } from '.'

// Real prompt hooks, cache, ownership router, and dispatcher; only the external
// gateway edge is substituted. No test injects a corrupted ownership mapping.
vi.mock('@/store/gateway', async original => ({
  ...(await original<Record<string, unknown>>()),
  requestGatewayForAgent: vi.fn(),
  requestGatewayForProfile: vi.fn(),
  retainGatewayForSessionTurn: vi.fn(async () => () => undefined)
}))

const busyRef = { current: false }
let routedStoredId: string | null = 'stored-B'
let handle: { actions: ReturnType<typeof usePromptActions>; cache: ReturnType<typeof useSessionStateCache> }

function Harness() {
  const activeSessionId = useStore($activeSessionId)
  const selectedStoredSessionId = useStore($selectedStoredSessionId)

  const cache = useSessionStateCache({
    activeSessionId,
    selectedStoredSessionId,
    busyRef,
    setAwaitingResponse,
    setBusy,
    setMessages
  })

  const requestGateway = createSessionRpcDispatcher({
    ...cache,
    ambientRequest: async () => {
      throw new Error('unexpected ambient request')
    }
  })

  const actions = usePromptActions({
    activeSessionId,
    ...cache,
    busyRef,
    branchCurrentSession: async () => false,
    createBackendSessionForSend: async () => {
      throw new Error('unexpected create')
    },
    getRoutedStoredSessionId: () => routedStoredId,
    getRouteToken: () => `${routedStoredId ? sessionRoute(routedStoredId) : '/'}::`,
    handleSkinCommand: () => '',
    openMemoryGraph: () => undefined,
    refreshSessions: async () => undefined,
    requestGateway,
    resumeStoredSession: async () => {
      throw new Error('unexpected foreground resume')
    },
    startFreshSessionDraft: () => undefined,
    sttEnabled: false
  })

  handle = { actions, cache }

  return null
}

function seed() {
  routedStoredId = 'stored-B'
  busyRef.current = false
  // Same profile name on distinct backends: profile equality is not ownership.
  setSessions(
    ['A', 'B'].map(id => ({
      id: `stored-${id}`,
      connection_id: `connection-${id}`,
      profile: 'default',
      source: 'desktop',
      message_count: 1
    })) as SessionInfo[]
  )
  setSelectedStoredSessionId('stored-B')
  setActiveSessionId('rt-B')
  render(<Harness />)
  act(() => {
    for (const id of ['A', 'B']) {
      handle.cache.updateSessionState(
        `rt-${id}`,
        state => ({
          ...state,
          messages: [{ id: `history-${id}`, role: 'assistant', parts: [textPart(`history ${id}`)] }]
        }),
        `stored-${id}`
      )
    }
  })
}

afterEach(() => {
  cleanup()
  clearAllSessionStates()
  clearSingleFlightSessionResumeState()
  setActiveSessionId(null)
  setActiveSessionStoredIdRotation(null)
  setSelectedStoredSessionId(null)
  setSessions([])
  setBusy(false)
  setAwaitingResponse(false)
  setMessages([])
  window.localStorage.clear()
  vi.clearAllMocks()
})

// #141644: a pane whose busy belief outlived its turn sends the next message as a correction. The
// owner refuses it before any agent sees it; the action must hand the words back (false) so the
// composer re-queues them, never throw (which discarded the message with a "prompt failed" toast).
it.each(['execution_not_ready', 'stale_generation'])(
  'redirectPrompt hands back a correction the owner refused undelivered (%s)',
  async reason => {
    vi.mocked(requestGatewayForAgent).mockRejectedValue(
      new JsonRpcGatewayError(reason, { code: 4001, data: { reason } })
    )
    seed()

    await act(async () => {
      expect(await handle.actions.redirectPrompt('next message')).toBe(false)
    })

    expect(requestGatewayForAgent).toHaveBeenCalledTimes(1)
    const messages = handle.cache.sessionStateByRuntimeIdRef.current.get('rt-B')!.messages
    expect(messages.map(message => message.id)).toEqual(['history-B'])
  }
)

it('redirectPrompt still surfaces a refusal that may have reached the turn', async () => {
  vi.mocked(requestGatewayForAgent).mockRejectedValue(
    new JsonRpcGatewayError('permission_denied', { code: 4001, data: { reason: 'permission_denied' } })
  )
  seed()

  await act(async () => {
    await expect(handle.actions.redirectPrompt('next message')).rejects.toThrow('permission_denied')
  })
})

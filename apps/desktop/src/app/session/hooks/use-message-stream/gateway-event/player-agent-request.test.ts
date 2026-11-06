import { afterEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { $sessionTiles } from '@/store/session-states'

import { playerAgentScopeFromServerRequest } from './player-agent-request'
import { handleServerRequest, type ServerRequestContext } from './server-requests'

const linked = vi.hoisted(() => vi.fn(() => false))
const dispatch = vi.hoisted(() => vi.fn(async () => ({ operation: 'context' })))
vi.mock('@/lib/player-agent-bridge', () => ({
  isPlayerAgentScopeLinked: linked,
  handlePlayerAgentServerRequest: dispatch
}))

const deps = {
  activeSessionIdRef: { current: 'runtime-a' },
  sessionInterrupted: () => false,
  sessionStateByRuntimeIdRef: { current: new Map() },
  updateSessionState: (_id: string, update: (s: ReturnType<typeof createClientSessionState>) => unknown) =>
    update(createClientSessionState('stored-a')),
  upsertToolCall: () => undefined
} as ServerRequestContext['deps']

function deliver(
  active: string,
  params: Record<string, unknown> = { session_id: 'runtime-a', operation: { type: 'context' } }
) {
  const request = {
    id: 'request-1',
    method: 'player.agent.request',
    connectionId: 'remote-a',
    profile: 'profile-a',
    params,
    respond: vi.fn(),
    fail: vi.fn(),
    decline: vi.fn()
  }

  expect(handleServerRequest(request, deps, active)).toBe(true)

  return request
}

afterEach(() => {
  vi.clearAllMocks()
  linked.mockReturnValue(false)
  $sessionTiles.set([])
})
describe('linked native player request ownership', () => {
  it('declines in another window and in an unlinked window showing the same chat', () => {
    const elsewhere = deliver('runtime-b'),
      sameChat = deliver('runtime-a')

    for (const request of [elsewhere, sameChat]) {
      expect(request.decline).toHaveBeenCalledOnce()
      expect(request.respond).not.toHaveBeenCalled()
    }

    expect(dispatch).not.toHaveBeenCalled()
  })
  it('uses linked transport tags rather than route claims in the operation payload', async () => {
    linked.mockReturnValue(true)
    const operation = { type: 'context', profileId: 'model-claim' }
    const request = deliver('runtime-a', { session_id: 'runtime-a', operation })
    await vi.waitFor(() => expect(request.respond).toHaveBeenCalledOnce())
    expect(dispatch).toHaveBeenCalledWith(
      { connectionId: 'remote-a', profileId: 'profile-a', sessionId: 'runtime-a' },
      operation
    )
    expect(request.decline).not.toHaveBeenCalled()
    expect(playerAgentScopeFromServerRequest({ profile: '  ', connectionId: undefined }, 'runtime-a')).toEqual({
      connectionId: 'local',
      profileId: 'default',
      sessionId: 'runtime-a'
    })
  })
  it('refuses an unscoped request without sharing player context', () => {
    linked.mockReturnValue(true)
    const request = deliver('runtime-a', { operation: { type: 'context' } })
    expect(request.decline).toHaveBeenCalledOnce()
    expect(dispatch).not.toHaveBeenCalled()
  })
})

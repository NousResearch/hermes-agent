import { registryBackendScopeKey } from '@hermes/shared'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { $unreadFinishedSessionIds } from '@/store/session'
import {
  $sessionStates,
  $workingSessionIds,
  clearAllSessionStates,
  liveSessionScopes,
  publishSessionState,
  recordSessionEventScope
} from '@/store/session-states'

import { wipeSessionListsForGatewaySwitch } from './gateway-switch'

vi.mock('@/lib/query-client', () => ({
  invalidateProfileScopedQueries: vi.fn()
}))

vi.mock(import('@/store/gateway'), async importOriginal => ({
  ...(await importOriginal()),
  isPooledRegistryRoute: vi.fn()
}))

const { isPooledRegistryRoute } = await import('@/store/gateway')

// The incident: a turn streaming on the pooled `local` registry route while
// the window switches to a remote source.
const LOCAL_SCOPE = registryBackendScopeKey('local', 'default')

function startTurn(runtimeId: string, storedSessionId: string, source?: { connectionId: string }) {
  if (source) {
    recordSessionEventScope({ connectionId: source.connectionId, profile: 'default', session_id: runtimeId })
  }

  publishSessionState(runtimeId, { ...createClientSessionState(storedSessionId), busy: true })
}

describe('connection switch keeps live work on the source it leaves', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    localStorage.clear()
    clearAllSessionStates()
    $unreadFinishedSessionIds.set([])
    vi.mocked(isPooledRegistryRoute).mockImplementation(scope => scope === LOCAL_SCOPE)
  })

  afterEach(() => {
    clearAllSessionStates()
    $unreadFinishedSessionIds.set([])
    localStorage.clear()
    vi.useRealTimers()
  })

  it('keeps the turn claiming its route through the switch, and its finish lights the unread dot', () => {
    startTurn('173890a7', '20261008_195234_dd1c54', { connectionId: 'local' })

    wipeSessionListsForGatewaySwitch()

    // The claim is what keeps the pruner off the outgoing socket: without it
    // the socket closed mid-turn and the backend detached the runtime.
    expect(liveSessionScopes()).toContain(LOCAL_SCOPE)
    expect($workingSessionIds.get()).toContain('20261008_195234_dd1c54')

    // The turn's terminal event still arrives over the kept route.
    publishSessionState('173890a7', { ...$sessionStates.get()['173890a7']!, busy: false })

    expect($workingSessionIds.get()).not.toContain('20261008_195234_dd1c54')
    expect($unreadFinishedSessionIds.get()).toContain('20261008_195234_dd1c54')
  })

  it('still wipes state the switch retires: the primary, unpooled routes, idle and unbound runtimes', () => {
    startTurn('primary-rt', 'stored-primary')
    startTurn('gone-rt', 'stored-gone', { connectionId: 'retired' })
    startTurn('draft-rt', '', { connectionId: 'local' })
    recordSessionEventScope({ connectionId: 'local', profile: 'default', session_id: 'idle-rt' })
    publishSessionState('idle-rt', createClientSessionState('stored-idle'))

    wipeSessionListsForGatewaySwitch()

    expect($sessionStates.get()).toEqual({})
    expect(liveSessionScopes().size).toBe(0)
  })
})

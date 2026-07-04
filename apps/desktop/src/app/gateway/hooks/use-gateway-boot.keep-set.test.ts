import { afterEach, describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/hermes'
import { $attentionSessionIds, $workingSessionIds, setSessions, setSessionWorking } from '@/store/session'
import { $splitPaneSession } from '@/store/split'

import { keptGatewayProfiles } from './use-gateway-boot'

// The prune keep-set (design §4.4 / §6 step 13): an idle split session's
// profile must keep its background socket, or the pane's next stream/steer
// dies with a reaped backend.

function storedSession(overrides: Partial<SessionInfo> = {}): SessionInfo {
  return {
    ended_at: null,
    id: 'stored-1',
    input_tokens: 0,
    is_active: false,
    last_active: 1,
    message_count: 0,
    model: null,
    output_tokens: 0,
    preview: null,
    source: 'desktop',
    started_at: 1,
    title: 'stored',
    tool_call_count: 0,
    ...overrides
  }
}

function resetStores() {
  for (const id of $workingSessionIds.get()) {
    setSessionWorking(id, false)
  }

  $attentionSessionIds.set([])
  setSessions([])
  $splitPaneSession.set(null)
}

describe('keptGatewayProfiles', () => {
  afterEach(resetStores)

  it('keeps profiles with a running session (baseline behavior unchanged)', () => {
    setSessions([storedSession({ id: 'stored-a', profile: 'coder' })])
    setSessionWorking('stored-a', true)

    expect(keptGatewayProfiles()).toEqual(new Set(['coder']))
  })

  it("includes the split pane's session profile even while that session is idle", () => {
    setSessions([storedSession({ id: 'stored-a', profile: 'coder' })])
    setSessionWorking('stored-a', true)
    $splitPaneSession.set({ profile: 'analyst', storedId: 'stored-split' })

    expect(keptGatewayProfiles()).toEqual(new Set(['coder', 'analyst']))
  })

  it("normalizes an untagged split session to the default profile's key", () => {
    $splitPaneSession.set({ profile: null, storedId: 'stored-split' })

    expect(keptGatewayProfiles()).toEqual(new Set(['default']))
  })

  it('drops the split profile once the split closes (no leaked keep-alive)', () => {
    $splitPaneSession.set({ profile: 'analyst', storedId: 'stored-split' })
    expect(keptGatewayProfiles()).toEqual(new Set(['analyst']))

    $splitPaneSession.set(null)
    expect(keptGatewayProfiles()).toEqual(new Set())
  })
})

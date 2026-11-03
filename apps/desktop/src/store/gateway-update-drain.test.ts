import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  $gatewayUpdateDrains,
  bindGatewayUpdateDrain,
  clearGatewayUpdateDrain,
  drainOwnerKey,
  gatewayUpdateDrainFor,
  refreshGatewayUpdateDrain,
  sessionUpdateDrain,
  updateDrainTurns
} from './gateway-update-drain'
import { $sessionControlBySession } from './session-control'

// Two chats on two gateways: s-a lives on connection A, s-b on connection B.
const owners: Record<string, { connectionId: string; profile: string }> = {
  's-a': { connectionId: 'conn-a', profile: 'default' },
  's-b': { connectionId: 'conn-b', profile: 'default' }
}

const describedBy: Record<string, unknown> = {}

vi.mock('./session-states', () => ({
  knownOwnerForSession: (sessionId: string) => owners[sessionId],
  requestForOwnedSession: async (sessionId: string) => describedBy[owners[sessionId].connectionId]
}))

const drain = { reason: 'update' as const, target: 'v0.99.0', waiting_for: { chat: 1, cron: 2 } }

describe('gateway update drain', () => {
  let unbind = () => {}

  afterEach(() => {
    unbind()
    $gatewayUpdateDrains.set({})
    $sessionControlBySession.set({})
  })

  it('asks the gateway that refused, and keeps its cause off the other gateway', async () => {
    describedBy['conn-a'] = { drain: null }
    describedBy['conn-b'] = { drain }
    // The ambient (focused-tile) requester is never the one asked for a background tile's refusal.
    unbind = bindGatewayUpdateDrain(async () => Promise.reject(new Error('asked the focused gateway')))
    $sessionControlBySession.set({ 's-b': { error: 'runtime_draining' } as never })
    await vi.waitFor(() => expect(sessionUpdateDrain($gatewayUpdateDrains.get(), 's-b')).toEqual(drain))
    expect(sessionUpdateDrain($gatewayUpdateDrains.get(), 's-a')).toBeNull()
    expect(updateDrainTurns(drain)).toBe(3)
  })

  it('keeps the cause while the drained owner is gone, clears it when a new owner answers, and drops a stale reply', async () => {
    const key = drainOwnerKey('default')
    let answer: () => Promise<unknown> = async () => ({ drain })
    const describe = () => answer() as Promise<{ drain: typeof drain | null }>
    await refreshGatewayUpdateDrain(key, describe)
    answer = async () => Promise.reject(new Error('Hermes gateway unavailable'))
    await refreshGatewayUpdateDrain(key, describe)
    expect(gatewayUpdateDrainFor($gatewayUpdateDrains.get(), key)).toEqual(drain)
    answer = async () => ({ drain: null })
    await refreshGatewayUpdateDrain(key, describe)
    expect(gatewayUpdateDrainFor($gatewayUpdateDrains.get(), key)).toBeNull()

    let late: (value: unknown) => void = () => {}
    answer = () => new Promise(resolve => (late = resolve))
    const pending = refreshGatewayUpdateDrain(key, describe)
    clearGatewayUpdateDrain(key)
    late({ drain })
    await pending
    expect(gatewayUpdateDrainFor($gatewayUpdateDrains.get(), key)).toBeNull()
  })
})

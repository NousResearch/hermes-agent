/**
 * Command Center M1 — provider and data-layer contract tests.
 *
 * Verifies the Architect Path B correction:
 *   1. The provider (./provider.ts) is pure/in-process — no network primitive
 *      is reachable from it, and it never throws for any exported function.
 *   2. ./api.ts calls the local provider directly and produces the same
 *      Envelope<T> shape the screens expect, with zero network calls.
 *   3. Runtime proof: globalThis.fetch / XMLHttpRequest are never invoked by
 *      any fetch* function in api.ts, even when installed as throwing spies —
 *      if a hidden network path existed, these tests would fail loudly.
 *   4. Truthful unavailable/unknown/synthetic state assertions per Architect/
 *      Picasso guidance: orb is always 'unknown', knowledge is always
 *      'unavailable', every envelope's source is 'synthetic-m1'.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  ccKeys,
  fetchAgents,
  fetchCapabilities,
  fetchKnowledge,
  fetchOrbState,
  fetchSecurity,
  fetchSummary,
  fetchWork,
} from './api'
import * as provider from './provider'

// ---------------------------------------------------------------------------
// Network-seam tripwire: install throwing spies on every network primitive.
// If any code path in api.ts/provider.ts calls one, the relevant test fails
// with a clear "network call attempted" error instead of silently passing.
// ---------------------------------------------------------------------------
function installNetworkTripwire() {
  const fetchSpy = vi.fn(() => {
    throw new Error('network call attempted: fetch()')
  })

  const xhrSpy = vi.fn(() => {
    throw new Error('network call attempted: XMLHttpRequest')
  })

  vi.stubGlobal('fetch', fetchSpy)
  vi.stubGlobal('XMLHttpRequest', xhrSpy)

  return { fetchSpy, xhrSpy }
}

beforeEach(() => {
  vi.unstubAllGlobals()
  vi.clearAllMocks()
})

describe('network isolation (Architect Path B contract)', () => {
  it('fetchCapabilities never touches fetch or XMLHttpRequest', async () => {
    const { fetchSpy, xhrSpy } = installNetworkTripwire()
    const env = await fetchCapabilities()

    expect(fetchSpy).not.toHaveBeenCalled()
    expect(xhrSpy).not.toHaveBeenCalled()
    expect(env.source).toBe('synthetic-m1')
  })

  it.each([
    ['fetchSummary', fetchSummary],
    ['fetchOrbState', fetchOrbState],
    ['fetchAgents', fetchAgents],
    ['fetchWork', fetchWork],
    ['fetchKnowledge', fetchKnowledge],
    ['fetchSecurity', fetchSecurity],
  ] as const)('%s never touches fetch or XMLHttpRequest', async (_name, fn) => {
    const { fetchSpy, xhrSpy } = installNetworkTripwire()
    const env = await fn()

    expect(fetchSpy).not.toHaveBeenCalled()
    expect(xhrSpy).not.toHaveBeenCalled()
    expect(env.source).toBe('synthetic-m1')
  })

  it('the api module never imports ctx.rest or any @hermes/plugin-sdk rest binding', async () => {
    // If api.ts ever re-adds a `bindApi`/`rest` binding, this import would
    // resolve to a function; M1's contract is that no such export exists.
    const mod = (await import('./api')) as Record<string, unknown>
    expect(mod.bindApi).toBeUndefined()
    expect(mod.rest).toBeUndefined()
  })
})

describe('provider determinism and contract shape', () => {
  it('every provider function returns a stable-shaped envelope', () => {
    const clock = () => '2026-01-01T00:00:00.000Z'

    const capabilities = provider.getCapabilities(clock)
    const summary = provider.getSummary(clock)
    const orb = provider.getOrbState(clock)
    const agents = provider.getAgents(clock)
    const work = provider.getWork(clock)
    const knowledge = provider.getKnowledge(clock)
    const security = provider.getSecurity(clock)

    for (const env of [capabilities, summary, orb, agents, work, knowledge, security]) {
      expect(env.contract_version).toBe(provider.CONTRACT_VERSION)
      expect(env.source).toBe('synthetic-m1')
      expect(env.generated_at).toBe('2026-01-01T00:00:00.000Z')
      expect(Array.isArray(env.warnings)).toBe(true)
    }
  })

  it('is deterministic given a fixed clock: two calls produce identical output', () => {
    const clock = () => '2026-01-01T00:00:00.000Z'

    const a = provider.getWork(clock)
    const b = provider.getWork(clock)

    expect(a).toEqual(b)
  })

  it('orb state is always unknown — no health inference or trigger logic', () => {
    const env = provider.getOrbState()

    expect(env.data.state).toBe('unknown')
    expect(env.data.visual_hint).toBe('neutral')
    expect(env.data.aria_label.toLowerCase()).toContain('unknown')
  })

  it('knowledge status is always unavailable with a gate note, never real Vault data', () => {
    const env = provider.getKnowledge()

    expect(env.data.state).toBe('unavailable')
    expect(env.status).toBe('unavailable')
    expect(env.data.gate_note).toBeTruthy()
  })

  it('capabilities report command execution and all real-integration features as disabled', () => {
    const env = provider.getCapabilities()

    expect(env.data.features.command_execution).toBe(false)
    expect(env.data.features.real_agent_data).toBe(false)
    expect(env.data.features.real_knowledge_data).toBe(false)
    expect(env.data.features.real_security_data).toBe(false)
    expect(env.data.policy_state.command_execution).toBe('disabled')
    expect(env.data.policy_state.real_integration).toBe('not_authorized')
  })

  it('agents include both live and proposed implementation statuses, honestly labeled', () => {
    const env = provider.getAgents()
    const statuses = new Set(env.data.agents.map(a => a.implementation_status))

    expect(statuses.has('live')).toBe(true)
    expect(statuses.has('proposed')).toBe(true)

    // Proposed agents must not carry a live_status — no capability is implied.
    for (const agent of env.data.agents.filter(a => a.implementation_status === 'proposed')) {
      expect(agent.live_status).toBeNull()
    }
  })

  it('security events use the what / why-it-matters / what-is-needed shape', () => {
    const env = provider.getSecurity()

    for (const event of env.data.events) {
      expect(event.what).toBeTruthy()
      expect(event.why_it_matters).toBeTruthy()
      expect(event.what_is_needed).toBeTruthy()
    }
  })
})

describe('query key stability', () => {
  it('exposes one stable key per screen data need', () => {
    expect(Object.keys(ccKeys).sort()).toEqual(
      ['agents', 'capabilities', 'knowledge', 'orbState', 'security', 'summary', 'work'].sort()
    )
  })
})

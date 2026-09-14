import { describe, expect, it } from 'vitest'

import { setupRuntimeCheckResult, setupStatusResult } from '@/test/contract'
import { gatewayRequestMock } from '@/test/gateway-request'

import {
  evaluateRuntimeReadiness,
  fetchRuntimeReadinessSignals,
  interpretRuntimeReadiness,
  runtimeReadinessDisplay
} from './runtime-readiness'

describe('interpretRuntimeReadiness', () => {
  it('prefers runtime_check when both signals exist', () => {
    const result = interpretRuntimeReadiness({
      setup: setupStatusResult({ provider_configured: false }),
      setupError: null,
      runtime: setupRuntimeCheckResult({ ok: true }),
      runtimeError: null
    })

    expect(result).toEqual({
      checksDisagree: true,
      ready: true,
      reason: null,
      source: 'runtime_check'
    })
  })

  it('surfaces runtime mismatch details when runtime_check fails', () => {
    const result = interpretRuntimeReadiness({
      setup: setupStatusResult({ provider_configured: true }),
      setupError: null,
      runtime: setupRuntimeCheckResult({ error: 'No provider can serve the selected model.', ok: false }),
      runtimeError: null
    })

    expect(result.ready).toBe(false)
    expect(result.source).toBe('runtime_check')
    expect(result.checksDisagree).toBe(true)
    expect(result.reason).toContain('No provider can serve the selected model.')
    expect(result.reason).toContain('setup.status reports configured credentials')
  })

  it('falls back to setup.status when runtime_check has no boolean result', () => {
    const result = interpretRuntimeReadiness({
      setup: setupStatusResult({ provider_configured: true }),
      setupError: null,
      runtime: null,
      runtimeError: 'runtime check RPC unavailable'
    })

    expect(result).toEqual({
      checksDisagree: false,
      ready: true,
      reason: null,
      source: 'setup_status'
    })
  })

  it('uses explicit fallback when both checks are missing', () => {
    const result = interpretRuntimeReadiness({
      setup: null,
      setupError: 'setup.status timeout',
      runtime: null,
      runtimeError: 'setup.runtime_check timeout'
    })

    expect(result.ready).toBe(false)
    expect(result.source).toBe('fallback')
    expect(result.reason).toBe('setup.runtime_check timeout')
  })
})

describe('fetchRuntimeReadinessSignals', () => {
  it('scopes setup.runtime_check to the requested provider', async () => {
    const requestGateway = gatewayRequestMock({
      'setup.runtime_check': () => setupRuntimeCheckResult({ ok: true }),
      'setup.status': () => setupStatusResult({ provider_configured: true })
    })

    await fetchRuntimeReadinessSignals(requestGateway, 'nous')

    expect(requestGateway.mock.calls.map(([method, params]) => ({ method, params }))).toEqual([
      { method: 'setup.status', params: {} },
      { method: 'setup.runtime_check', params: { provider: 'nous' } }
    ])
  })
})

describe('evaluateRuntimeReadiness', () => {
  it('forwards requestedProvider to setup.runtime_check', async () => {
    const requestGateway = gatewayRequestMock({
      'setup.runtime_check': params => {
        expect(params).toEqual({ provider: 'nous' })

        return setupRuntimeCheckResult({ ok: true })
      },
      'setup.status': () => setupStatusResult({ provider_configured: true })
    })

    const result = await evaluateRuntimeReadiness(requestGateway, { requestedProvider: 'nous' })

    expect(result.ready).toBe(true)
  })
})

describe('runtimeReadinessDisplay', () => {
  it('does not call configured credentials setup when runtime resolution fails', () => {
    expect(
      runtimeReadinessDisplay({
        checksDisagree: true,
        ready: false,
        reason: 'Anthropic cannot serve the selected model.',
        source: 'runtime_check'
      })
    ).toBe('unavailable')
  })

  it('keeps needs-setup for an authoritative unconfigured result', () => {
    expect(
      runtimeReadinessDisplay({
        checksDisagree: false,
        ready: false,
        reason: 'No provider configured.',
        source: 'setup_status'
      })
    ).toBe('needs_setup')
  })
})

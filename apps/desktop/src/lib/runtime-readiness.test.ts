import { describe, expect, it } from 'vitest'

import { evaluateRuntimeReadiness, fetchRuntimeReadinessSignals, interpretRuntimeReadiness } from './runtime-readiness'

describe('interpretRuntimeReadiness', () => {
  it('prefers runtime_check when both signals exist', () => {
    const result = interpretRuntimeReadiness({
      setup: { provider_configured: false },
      setupError: null,
      runtime: { ok: true },
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
      setup: { provider_configured: true },
      setupError: null,
      runtime: { error: 'No provider can serve the selected model.', ok: false },
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
      setup: { provider_configured: true },
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
    const calls: Array<{ method: string; params?: Record<string, unknown> }> = []

    const requestGateway = async <T = unknown>(method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'setup.status') {
        return { provider_configured: true } as T
      }

      if (method === 'setup.runtime_check') {
        return { ok: true } as T
      }

      throw new Error(`unexpected method: ${method}`)
    }

    await fetchRuntimeReadinessSignals(requestGateway, 'nous')

    expect(calls).toEqual([{ method: 'setup.status' }, { method: 'setup.runtime_check', params: { provider: 'nous' } }])
  })

  it('forwards profile parameter to both setup.status and setup.runtime_check', async () => {
    const calls: Array<{ method: string; params?: Record<string, unknown> }> = []

    const requestGateway = async <T = unknown>(method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'setup.status') {
        return { provider_configured: true } as T
      }

      if (method === 'setup.runtime_check') {
        return { ok: true } as T
      }

      throw new Error(`unexpected method: ${method}`)
    }

    await fetchRuntimeReadinessSignals(requestGateway, 'nous', 'work')

    expect(calls).toEqual([
      { method: 'setup.status', params: { profile: 'work' } },
      { method: 'setup.runtime_check', params: { provider: 'nous', profile: 'work' } }
    ])
  })
})

describe('evaluateRuntimeReadiness', () => {
  it('forwards requestedProvider to setup.runtime_check', async () => {
    const requestGateway = async <T = unknown>(method: string, params?: Record<string, unknown>) => {
      if (method === 'setup.status') {
        return { provider_configured: true } as T
      }

      if (method === 'setup.runtime_check') {
        expect(params).toEqual({ provider: 'nous' })

        return { ok: true } as T
      }

      throw new Error(`unexpected method: ${method}`)
    }

    const result = await evaluateRuntimeReadiness(requestGateway, { requestedProvider: 'nous' })

    expect(result.ready).toBe(true)
  })

  it('forwards profile to setup.status and setup.runtime_check via options', async () => {
    const calls: Array<{ method: string; params?: Record<string, unknown> }> = []

    const requestGateway = async <T = unknown>(method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'setup.status') {
        return { provider_configured: true } as T
      }

      if (method === 'setup.runtime_check') {
        return { ok: true } as T
      }

      throw new Error(`unexpected method: ${method}`)
    }

    const result = await evaluateRuntimeReadiness(requestGateway, { requestedProvider: 'openai', profile: 'dev' })

    expect(result.ready).toBe(true)
    expect(calls).toEqual([
      { method: 'setup.status', params: { profile: 'dev' } },
      { method: 'setup.runtime_check', params: { provider: 'openai', profile: 'dev' } }
    ])
  })
})

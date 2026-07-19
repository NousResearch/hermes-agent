import { describe, expect, it } from 'vitest'

import {
  evaluateRuntimeReadiness,
  fetchRuntimeReadinessSignals,
  interpretRuntimeReadiness,
  requiresProfileIdentity
} from './runtime-readiness'

describe('requiresProfileIdentity', () => {
  it('requires identity only for app-global remote connections', () => {
    expect(requiresProfileIdentity({ mode: 'remote', source: 'settings' })).toBe(true)
    expect(requiresProfileIdentity({ mode: 'remote', source: 'env' })).toBe(true)
    expect(requiresProfileIdentity({ mode: 'remote', source: 'profile' })).toBe(false)
    expect(requiresProfileIdentity({ mode: 'local', source: 'local' })).toBe(false)
    expect(requiresProfileIdentity(null)).toBe(false)
  })
})

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

  it('preserves the literal default profile for a shared remote gateway', async () => {
    const calls: Array<{ method: string; params?: Record<string, unknown> }> = []

    const requestGateway = async <T = unknown>(method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      return (
        method === 'setup.status'
          ? { profile_name: 'default', provider_configured: true }
          : { ok: true, profile_name: 'default' }
      ) as T
    }

    await fetchRuntimeReadinessSignals(requestGateway, undefined, 'default')

    expect(calls).toEqual([
      { method: 'setup.status', params: { profile: 'default' } },
      { method: 'setup.runtime_check', params: { profile: 'default' } }
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

  it('scopes both readiness checks to the active profile', async () => {
    const calls: Array<{ method: string; params?: Record<string, unknown> }> = []

    const requestGateway = async <T = unknown>(method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      return (method === 'setup.status' ? { provider_configured: true } : { ok: true }) as T
    }

    await evaluateRuntimeReadiness(requestGateway, {
      profile: 'coder',
      requireProfileIdentity: true,
      requestedProvider: 'openai-codex'
    })

    expect(calls).toEqual([
      { method: 'setup.status', params: { profile: 'coder' } },
      { method: 'setup.runtime_check', params: { profile: 'coder', provider: 'openai-codex' } }
    ])
  })

  it('does not send a Desktop-local profile label to a dedicated backend', async () => {
    const calls: Array<{ method: string; params?: Record<string, unknown> }> = []

    const requestGateway = async <T = unknown>(method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      return (method === 'setup.status' ? { provider_configured: true } : { ok: true }) as T
    }

    const result = await evaluateRuntimeReadiness(requestGateway, {
      profile: 'desktop-only-label',
      requireProfileIdentity: false,
      requestedProvider: 'openai-codex'
    })

    expect(result.ready).toBe(true)
    expect(calls).toEqual([
      { method: 'setup.status', params: undefined },
      { method: 'setup.runtime_check', params: { provider: 'openai-codex' } }
    ])
  })

  it('rejects an old shared remote backend that cannot confirm the requested profile', async () => {
    const requestGateway = async <T = unknown>(method: string) =>
      (method === 'setup.status' ? { provider_configured: true } : { ok: true }) as T

    const result = await evaluateRuntimeReadiness(requestGateway, {
      profile: 'coder',
      requireProfileIdentity: true
    })

    expect(result.ready).toBe(false)
    expect(result.source).toBe('fallback')
    expect(result.reason).toMatch(/update/i)
  })

  it('rejects readiness resolved for a different remote profile', async () => {
    const requestGateway = async <T = unknown>(method: string) =>
      (method === 'setup.status'
        ? { profile_name: 'default', provider_configured: true }
        : { ok: true, profile_name: 'default' }) as T

    const result = await evaluateRuntimeReadiness(requestGateway, {
      profile: 'coder',
      requireProfileIdentity: true
    })

    expect(result.ready).toBe(false)
    expect(result.reason).toContain('default')
    expect(result.reason).toContain('coder')
  })

  it('accepts legacy readiness without a profile marker when no shared remote scope is required', async () => {
    const requestGateway = async <T = unknown>(method: string) =>
      (method === 'setup.status' ? { provider_configured: true } : { ok: true }) as T

    const result = await evaluateRuntimeReadiness(requestGateway, {
      profile: 'coder',
      requireProfileIdentity: false
    })

    expect(result.ready).toBe(true)
  })
})

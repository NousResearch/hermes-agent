import { describe, expect, it, vi } from 'vitest'

import {
  buildDashboardSessionUrl,
  type ContinueOnPhoneDependencies,
  resolveContinueOnPhoneUrl
} from './continue-on-phone'

function dependencies(
  overrides: Partial<ContinueOnPhoneDependencies> = {}
): ContinueOnPhoneDependencies {
  return {
    getRemoteAccess: vi.fn().mockResolvedValue({ public_url: 'https://hermes.example.com/agent' }),
    probe: vi.fn().mockResolvedValue({ authMode: 'oauth', reachable: true }),
    ...overrides
  }
}

describe('buildDashboardSessionUrl', () => {
  it('preserves a dashboard path prefix and scopes the resumed session', () => {
    expect(buildDashboardSessionUrl('https://hermes.example.com/agent', 'session / 42', 'work profile')).toBe(
      'https://hermes.example.com/agent/chat?resume=session+%2F+42&profile=work+profile'
    )
  })

  it('requires an HTTPS public URL', () => {
    expect(buildDashboardSessionUrl('http://hermes.example.com', 'session-42')).toBeNull()
    expect(buildDashboardSessionUrl('not a url', 'session-42')).toBeNull()
    expect(buildDashboardSessionUrl('https://hermes.example.com', '')).toBeNull()
  })

  it('never carries URL credentials into the handoff', () => {
    expect(buildDashboardSessionUrl('https://user:password@hermes.example.com', 'session-42')).toBeNull()
  })
})

describe('resolveContinueOnPhoneUrl', () => {
  it('returns an authenticated dashboard URL without embedding credentials', async () => {
    const deps = dependencies()

    const result = await resolveContinueOnPhoneUrl('session-42', 'work', deps)

    expect(result).toEqual({
      ok: true,
      url: 'https://hermes.example.com/agent/chat?resume=session-42&profile=work'
    })
    expect(deps.getRemoteAccess).toHaveBeenCalledWith('work')
    expect(deps.probe).toHaveBeenCalledWith('https://hermes.example.com/agent')
    expect(result.ok && result.url).not.toContain('token=')
  })

  it('refuses a dashboard without a configured public URL', async () => {
    const result = await resolveContinueOnPhoneUrl(
      'session-42',
      undefined,
      dependencies({ getRemoteAccess: vi.fn().mockResolvedValue({ public_url: '' }) })
    )

    expect(result).toEqual({ ok: false, reason: 'not-configured' })
  })

  it('refuses an insecure public URL before probing it', async () => {
    const deps = dependencies({
      getRemoteAccess: vi.fn().mockResolvedValue({ public_url: 'http://hermes.example.com' })
    })

    const result = await resolveContinueOnPhoneUrl('session-42', undefined, deps)

    expect(result).toEqual({ ok: false, reason: 'insecure-url' })
    expect(deps.probe).not.toHaveBeenCalled()
  })

  it('refuses an unreachable dashboard or one without the auth gate', async () => {
    const unreachable = await resolveContinueOnPhoneUrl(
      'session-42',
      undefined,
      dependencies({ probe: vi.fn().mockResolvedValue({ authMode: 'unknown', reachable: false }) })
    )

    const unauthenticated = await resolveContinueOnPhoneUrl(
      'session-42',
      undefined,
      dependencies({ probe: vi.fn().mockResolvedValue({ authMode: 'token', reachable: true }) })
    )

    expect(unreachable).toEqual({ ok: false, reason: 'unreachable' })
    expect(unauthenticated).toEqual({ ok: false, reason: 'auth-required' })
  })
})

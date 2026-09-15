import { afterEach, describe, expect, it } from 'vitest'

import { capabilityScoped, setApiRequestConnection, setApiRequestProfile } from '@/api/client'
import type { McpTestResult } from '@/hermes'

import {
  classifyProbe,
  freshProbe,
  NEEDS_AUTH_RE,
  PROBE_TTL_MS,
  probeCache,
  probeKey,
  resolveMcpOwner
} from './mcp-probe-cache'

const result = (over: Partial<McpTestResult> = {}): McpTestResult => ({ ok: true, tools: [], ...over })

afterEach(() => {
  setApiRequestConnection(null)
  setApiRequestProfile(null)
})

describe('classifyProbe', () => {
  it('classifies a successful probe as ok', () => {
    expect(classifyProbe(result())).toBe('ok')
  })

  it.each([
    'HTTP 401 Unauthorized',
    'invalid_token: The access token expired',
    'OAuth authorization required',
    'authentication failed'
  ])('classifies "%s" as needs-auth', error => {
    expect(classifyProbe(result({ ok: false, error }))).toBe('needs-auth')
  })

  it('classifies other failures as error', () => {
    expect(classifyProbe(result({ ok: false, error: 'ECONNREFUSED 127.0.0.1:3845' }))).toBe('error')
  })

  it('classifies a failure without an error string as error', () => {
    expect(classifyProbe(result({ ok: false }))).toBe('error')
  })
})

describe('probeKey', () => {
  it('scopes by profile, name, and connection-relevant config', () => {
    const server = { url: 'https://api.githubcopilot.com/mcp/' }
    expect(probeKey('github', server, 'default')).not.toBe(probeKey('github', server, 'work'))
    expect(probeKey('github', server, 'default')).not.toBe(probeKey('gh2', server, 'default'))
    expect(probeKey('github', server, 'default')).not.toBe(
      probeKey('github', { url: 'https://other.example/mcp' }, 'default')
    )
  })

  it('ignores non-connection fields so cosmetic edits still hit the cache', () => {
    const server = { url: 'https://api.example/mcp' }
    expect(probeKey('s', server, 'default')).toBe(probeKey('s', { ...server, description: 'hi' }, 'default'))
  })
})

describe('resolveMcpOwner', () => {
  it('shares the registered ambient owner between health and tab callers', () => {
    setApiRequestProfile('default')

    const healthOwner = resolveMcpOwner(undefined, 'default', 'connection-a', {
      baseUrl: 'https://a.example',
      mode: 'remote'
    })

    const tabOwner = resolveMcpOwner(undefined, 'default', 'connection-a', {
      baseUrl: 'https://a.example',
      mode: 'remote'
    })

    const server = { url: 'https://mcp.example.test/mcp' }

    const result = resultForTest()

    expect(healthOwner).toEqual(tabOwner)
    expect(healthOwner?.key).toBe('connection-a::default')
    probeCache.set(probeKey('shared', server, healthOwner!.key), { at: 1_000, result })
    expect(freshProbe(probeKey('shared', server, tabOwner!.key), 1_001)).toBe(result)
    expect(probeKey('shared', server, resolveMcpOwner(undefined, 'default', 'connection-b', null)!.key)).not.toBe(
      probeKey('shared', server, healthOwner!.key)
    )
    probeCache.clear()
  })

  it('uses a legacy descriptor identity without inventing a request pin', () => {
    const owner = resolveMcpOwner(undefined, 'default', null, {
      baseUrl: 'https://legacy.example',
      mode: 'remote'
    })

    expect(owner?.exact).toBe(false)
    expect(owner?.request).toEqual({})
    expect(owner?.key).toContain('remote.')
  })

  it('honors an explicit scope and fails closed without any owner descriptor', () => {
    expect(resolveMcpOwner({ connectionId: 'local', profile: 'default' }, 'other', 'remote', null)).toEqual({
      exact: true,
      key: 'local::default',
      request: { connectionId: 'local', profile: 'default' }
    })
    expect(
      resolveMcpOwner({ profile: 'primary' }, 'primary', 'remote', {
        baseUrl: 'https://ambient.example',
        mode: 'remote'
      })
    ).toBeNull()
    expect(resolveMcpOwner(undefined, 'default', null, null)).toBeNull()
  })

  it.each([
    { label: 'legacy string scope', scope: 'default' as const, ambientConnection: 'connection-a' },
    {
      label: 'explicit connection/profile scope',
      scope: { connectionId: 'connection-a', profile: 'default' } as const,
      ambientConnection: 'connection-b'
    }
  ])('keeps the $label request and owner key aligned', ({ scope, ambientConnection }) => {
    setApiRequestConnection(ambientConnection)
    setApiRequestProfile('default')

    const owner = resolveMcpOwner(scope, 'active-profile', ambientConnection, {
      baseUrl: `https://${ambientConnection}.example`,
      mode: 'remote'
    })

    expect(owner?.request).toEqual(capabilityScoped(scope))
    expect(owner?.key).toBe(`${scope === 'default' ? ambientConnection : 'connection-a'}::default`)
  })

  it.each([
    { label: 'explicit null profile', scope: { connectionId: 'remote', profile: null } },
    { label: 'explicit undefined profile', scope: { connectionId: 'remote', profile: undefined } },
    { label: 'explicit empty profile', scope: { connectionId: 'remote', profile: '' } },
    { label: 'null scope', scope: null },
    { label: 'empty string scope', scope: '' }
  ])('keeps $label request wire-equivalent to capabilityScoped', ({ scope }) => {
    setApiRequestConnection('ambient')
    setApiRequestProfile('ambient-profile')

    const owner = resolveMcpOwner(scope, 'active-profile', 'ambient', {
      baseUrl: 'https://ambient.example',
      mode: 'remote'
    })

    expect(owner?.request).toEqual(capabilityScoped(scope))
  })

  it('captures the ambient profile for an undefined scope', () => {
    setApiRequestConnection('ambient')
    setApiRequestProfile('ambient-profile')

    const owner = resolveMcpOwner(undefined, 'active-profile', 'ambient', {
      baseUrl: 'https://ambient.example',
      mode: 'remote'
    })

    expect(owner?.request).toEqual(capabilityScoped(undefined))
    expect(owner?.key).toBe('ambient::ambient-profile')
  })

  it('keeps an omitted ambient profile in a reserved primary slot', () => {
    setApiRequestConnection('ambient')
    setApiRequestProfile(null)

    const owner = resolveMcpOwner(undefined, 'active-profile', 'ambient', {
      baseUrl: 'https://ambient.example',
      mode: 'remote'
    })

    const literalDefault = resolveMcpOwner({ connectionId: 'ambient', profile: 'default' }, 'active-profile', 'other', null)

    expect(owner?.request).toEqual(capabilityScoped(undefined))
    expect(owner?.key).toBe('ambient::@primary-slot')
    expect(literalDefault?.key).toBe('ambient::default')
    expect(owner?.key).not.toBe(literalDefault?.key)
  })
})

describe('freshProbe', () => {
  it('returns a cached result inside the TTL and null after it', () => {
    const key = probeKey('ttl-test', { url: 'https://x' }, 'default')
    const cached = result()
    probeCache.set(key, { at: 1_000, result: cached })

    expect(freshProbe(key, 1_000 + PROBE_TTL_MS - 1)).toBe(cached)
    expect(freshProbe(key, 1_000 + PROBE_TTL_MS)).toBeNull()
    expect(freshProbe('missing', 0)).toBeNull()
    probeCache.delete(key)
  })
})

describe('NEEDS_AUTH_RE', () => {
  it('does not match unrelated failure text', () => {
    expect(NEEDS_AUTH_RE.test('connection timed out after 60000ms')).toBe(false)
  })
})

function resultForTest(): McpTestResult {
  return result({ tools: [{ name: 'tool', description: 'shared' }] })
}

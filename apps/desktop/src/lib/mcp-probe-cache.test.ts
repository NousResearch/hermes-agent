import { describe, expect, it } from 'vitest'

import type { McpTestResult } from '@/hermes'

import { classifyProbe, freshProbe, NEEDS_AUTH_RE, PROBE_TTL_MS, probeCache, probeKey } from './mcp-probe-cache'

const result = (over: Partial<McpTestResult> = {}): McpTestResult => ({ ok: true, tools: [], ...over })

describe('classifyProbe', () => {
  it('classifies a successful probe as ok', () => {
    expect(classifyProbe(result())).toBe('ok')
  })

  it.each([
    'HTTP 401 Unauthorized',
    'invalid_token: The access token expired',
    'OAuth authorization required',
    'authentication failed',
    'HTTP 403 Forbidden'
  ])('classifies "%s" as needs-auth', error => {
    expect(classifyProbe(result({ ok: false, error }))).toBe('needs-auth')
  })

  it.each([
    'ECONNREFUSED 127.0.0.1:3845',
    "Connecting to MCP server 'inspo' timed out after 30s (bounded by connect_timeout; an OAuth login also by oauth.timeout)",
    'The server responded, but no OAuth token was obtained — this provider may require a manually-registered OAuth client.'
  ])('classifies connectivity/non-auth OAuth prose as error: %s', error => {
    expect(classifyProbe(result({ ok: false, error }))).toBe('error')
  })

  it('prefers the backend failure class over prose', () => {
    expect(classifyProbe(result({ ok: false, error: 'connection failed', error_kind: 'auth' }))).toBe('needs-auth')
    expect(classifyProbe(result({ ok: false, error: 'OAuth login timed out', error_kind: 'network' }))).toBe('error')
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

import { describe, expect, it } from 'vitest'

import type { McpTestResult } from '@/hermes'

import { classifyProbe, freshProbe, PROBE_TTL_MS, probeCache, probeKey } from './mcp-probe-cache'

const result = (over: Partial<McpTestResult> = {}): McpTestResult => ({ ok: true, tools: [], ...over })

describe('classifyProbe', () => {
  it('classifies a successful probe as ok', () => {
    expect(classifyProbe(result())).toBe('ok')
  })

  it.each([
    'HTTP 401 Unauthorized',
    'invalid_token: The access token expired',
    'OAuth authorization required',
    'OAuth: authorization required',
    'authentication failed',
    'HTTP 403 Forbidden',
    "MCP OAuth for 'learn': non-interactive environment and no cached tokens found. Run `hermes mcp login learn` interactively first to complete initial authorization.",
    'MCP device authorization requires `hermes mcp login <server> --flow device`; background reconnects cannot start a device login',
    'MCP OAuth requires browser authorization but no interactive session is available (non-interactive/background context).',
    'OAuth callback requires an interactive session but none is available (non-interactive/background context)'
  ])('classifies "%s" as needs-auth', error => {
    expect(classifyProbe(result({ ok: false, error }))).toBe('needs-auth')
  })

  it.each([
    'ECONNREFUSED 127.0.0.1:3845',
    "MCP server 'rea' test timed out after 60s (bounded by connect_timeout; an OAuth login also by oauth.timeout)",
    'OAuth transport connection refused',
    'authentication timeout setting is 60s'
  ])('classifies non-auth failure "%s" as error', error => {
    expect(classifyProbe(result({ ok: false, error }))).toBe('error')
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

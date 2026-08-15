import { describe, expect, it } from 'vitest'

import { mcpServerFromToolName, REPAIR_RE } from './repair'

describe('mcpServerFromToolName', () => {
  it('extracts the server from a namespaced MCP tool', () => {
    expect(mcpServerFromToolName('mcp__figma__get_design_context')).toBe('figma')
    expect(mcpServerFromToolName('mcp__browsermcp__browser_navigate')).toBe('browsermcp')
  })

  it('handles multi-underscore server names', () => {
    expect(mcpServerFromToolName('mcp__comfy_cloud__generate')).toBe('comfy_cloud')
  })

  it('returns null for non-MCP tools', () => {
    expect(mcpServerFromToolName('terminal')).toBeNull()
    expect(mcpServerFromToolName('read_file')).toBeNull()
    expect(mcpServerFromToolName('mcp_not_namespaced')).toBeNull()
  })
})

describe('REPAIR_RE', () => {
  it.each([
    'HTTP 401 Unauthorized',
    'token has expired, please re-authenticate',
    'OAuth flow required',
    'authentication failed for server',
    'connection refused by remote host',
    'ECONNREFUSED 127.0.0.1:3845',
    'server disconnected before response'
  ])('flags auth/connection failures: %s', text => {
    expect(REPAIR_RE.test(text)).toBe(true)
  })

  it.each([
    'issue PROJ-123 not found',
    'no design nodes matched the query',
    'rate limit exceeded, retry after 60s',
    'invalid argument: node_id is required',
    'permission denied editing this file'
  ])('ignores tool-semantics failures: %s', text => {
    expect(REPAIR_RE.test(text)).toBe(false)
  })
})

import { describe, expect, it } from 'vitest'

import { mcpServerFromToolName } from './repair'

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

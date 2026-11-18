import { describe, expect, it } from 'vitest'

import { filterStdioSections } from './mcp-logs'

describe('filterStdioSections', () => {
  it('splits mcp-stderr.log by server for stamped and older banner lines', () => {
    const lines = [
      "===== [2026-09-27 09:00:00] starting MCP server 'fs' =====",
      'old fs output',
      "2026-09-28 13:18:46,062 ===== starting MCP server 'git' =====",
      '2026-09-28 13:18:46,100 git output',
      "2026-09-28 13:18:47,000 ===== starting MCP server 'fs' =====",
      '2026-09-28 13:18:47,050 new fs output'
    ]

    expect(filterStdioSections(lines, 'fs')).toEqual([lines[0], lines[1], lines[4], lines[5]])
    expect(filterStdioSections(lines, 'git')).toEqual([lines[2], lines[3]])
  })
})

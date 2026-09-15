import { describe, expect, it } from 'vitest'

import { estimateServerTokens, mcpServerUsagePrefix, sanitizeMcpNameComponent, serverUsageCount } from './mcp-cost'

describe('estimateServerTokens', () => {
  it('sums ceil(schema_chars / 4) over tools', () => {
    const tools = [
      { name: 'a', schema_chars: 400 },
      { name: 'b', schema_chars: 401 }
    ]

    // 100 + ceil(100.25) = 100 + 101
    expect(estimateServerTokens({}, tools)).toBe(201)
  })

  it('returns null when no tool carries schema_chars (older backend)', () => {
    expect(estimateServerTokens({}, [{ name: 'a' }, { name: 'b', description: 'x' }])).toBe(null)
    expect(estimateServerTokens({}, [])).toBe(null)
  })

  it('ignores invalid schema_chars values', () => {
    expect(
      estimateServerTokens({}, [
        { name: 'a', schema_chars: 0 },
        { name: 'b', schema_chars: -8 },
        { name: 'c', schema_chars: Number.NaN }
      ])
    ).toBe(null)
  })

  it('counts only ENABLED tools per the include/exclude filter', () => {
    const tools = [
      { name: 'keep', schema_chars: 40 },
      { name: 'drop', schema_chars: 4000 }
    ]

    expect(estimateServerTokens({ tools: { exclude: ['drop'] } }, tools)).toBe(10)
    expect(estimateServerTokens({ tools: { include: ['keep'] } }, tools)).toBe(10)
    // Everything filtered out → no estimate rather than a misleading 0.
    expect(estimateServerTokens({ tools: { include: ['other'] } }, tools)).toBe(null)
  })
})

describe('name mapping', () => {
  it('sanitizes like tools/mcp_tool.py sanitize_mcp_name_component', () => {
    expect(sanitizeMcpNameComponent('github-mcp')).toBe('github_mcp')
    expect(sanitizeMcpNameComponent('a.b c/d')).toBe('a_b_c_d')
    expect(sanitizeMcpNameComponent('plain_ok9')).toBe('plain_ok9')
  })

  it('builds the mcp__<server>__ registry prefix', () => {
    expect(mcpServerUsagePrefix('linear-app')).toBe('mcp__linear_app__')
  })
})

describe('serverUsageCount', () => {
  const calls = {
    mcp__linear_app__create_issue: 3,
    mcp__linear_app__list_issues: 2,
    mcp__linear_app__list_resources: 1,
    mcp__other__create_issue: 50,
    terminal: 900
  }

  it('sums counts across one server, matching sanitized prefix', () => {
    expect(serverUsageCount('linear-app', calls)).toBe(6)
    expect(serverUsageCount('other', calls)).toBe(50)
  })

  it('returns 0 for servers with no analytics rows', () => {
    expect(serverUsageCount('ghost', calls)).toBe(0)
    expect(serverUsageCount('linear-app', {})).toBe(0)
  })

  it('does not cross server boundaries on underscore-heavy names', () => {
    // `linear` must not swallow `linear_app`'s tools: the double-underscore
    // delimiter after the sanitized server name is part of the prefix.
    expect(serverUsageCount('linear', calls)).toBe(0)
  })
})

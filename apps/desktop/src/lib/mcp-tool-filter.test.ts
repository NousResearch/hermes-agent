import { describe, expect, it } from 'vitest'

import { countEnabledTools, isToolEnabled, readToolsFilter, toggleToolInServer } from './mcp-tool-filter'

describe('readToolsFilter', () => {
  it('returns empty when no tools object', () => {
    expect(readToolsFilter({ command: 'x' })).toEqual({ exclude: undefined, include: undefined })
  })

  it('reads include/exclude and ignores non-string entries', () => {
    expect(readToolsFilter({ tools: { exclude: ['c', 2], include: ['a', 'b', null] } })).toEqual({
      exclude: ['c'],
      include: ['a', 'b']
    })
  })
})

describe('isToolEnabled', () => {
  it('enables everything with no filter', () => {
    expect(isToolEnabled({ command: 'x' }, 'anything')).toBe(true)
  })

  it('include wins over exclude', () => {
    const server = { tools: { exclude: ['a'], include: ['a'] } }
    expect(isToolEnabled(server, 'a')).toBe(true)
    expect(isToolEnabled(server, 'b')).toBe(false)
  })

  it('exclude disables listed tools', () => {
    const server = { tools: { exclude: ['b'] } }
    expect(isToolEnabled(server, 'a')).toBe(true)
    expect(isToolEnabled(server, 'b')).toBe(false)
  })
})

describe('toggleToolInServer', () => {
  it('adds a fresh tool to a new exclude denylist when disabled', () => {
    const next = toggleToolInServer({ command: 'x' }, 'a')
    expect(next.tools).toEqual({ exclude: ['a'] })
  })

  it('re-enabling removes the tool and drops the empty exclude/tools', () => {
    const next = toggleToolInServer({ command: 'x', tools: { exclude: ['a'] } }, 'a')
    expect(next.tools).toBeUndefined()
  })

  it('respects include mode: toggling removes from include', () => {
    const next = toggleToolInServer({ tools: { include: ['a', 'b'] } }, 'a')
    expect(next.tools).toEqual({ include: ['b'] })
  })

  it('respects include mode: re-enabling adds back to include', () => {
    const next = toggleToolInServer({ tools: { include: ['b'] } }, 'a')
    expect(next.tools).toEqual({ include: ['b', 'a'] })
  })

  it('preserves sibling tools keys like resources/prompts', () => {
    const next = toggleToolInServer({ tools: { resources: false } }, 'a')
    expect(next.tools).toEqual({ exclude: ['a'], resources: false })
  })

  it('does not mutate the input server', () => {
    const server = { tools: { exclude: ['a'] } }
    toggleToolInServer(server, 'b')
    expect(server.tools.exclude).toEqual(['a'])
  })
})

describe('countEnabledTools', () => {
  it('counts enabled tools out of a discovered list', () => {
    const server = { tools: { exclude: ['b'] } }
    expect(countEnabledTools(server, ['a', 'b', 'c'])).toBe(2)
  })
})

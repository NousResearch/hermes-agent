import { createElement } from 'react'
import { describe, expect, it } from 'vitest'

import { countSelectItems, filterSelectChildren, selectItemMatchesQuery } from './select-search'

function item(value: string, label = value) {
  return createElement('div', { value, 'data-slot': 'select-item' }, label)
}

function group(...children: ReturnType<typeof item>[]) {
  return createElement('div', { 'data-slot': 'select-group' }, ...children)
}

describe('select-search', () => {
  it('matches model ids, labels, and folded punctuation', () => {
    const row = item('openai/gpt-5.6-luna-pro', 'GPT 5.6 Luna Pro')

    expect(selectItemMatchesQuery(row, 'luna')).toBe(true)
    expect(selectItemMatchesQuery(row, 'gpt-5.6')).toBe(true)
    expect(selectItemMatchesQuery(row, 'gpt 5 6 luna')).toBe(true)
    expect(selectItemMatchesQuery(row, 'claude')).toBe(false)
  })

  it('filters items and drops empty groups', () => {
    const tree = [
      group(item('openai/gpt-5.6-luna-pro'), item('anthropic/claude-sonnet-5')),
      item('openrouter/pareto-code')
    ]

    const filtered = filterSelectChildren(tree, 'luna')
    const remaining = createElement('div', null, filtered)

    expect(countSelectItems(remaining)).toBe(1)
    expect(selectItemMatchesQuery(item('openai/gpt-5.6-luna-pro'), 'luna')).toBe(true)
  })

  it('keeps the full list when the query is empty', () => {
    const tree = [item('a'), item('b')]

    expect(filterSelectChildren(tree, '   ')).toEqual(tree)
  })
})

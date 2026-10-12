import { describe, expect, it } from 'vitest'

import { compareAlphabetically, sortByLabel } from './alphabetical'

describe('compareAlphabetically', () => {
  it('is case-insensitive and orders numbers naturally', () => {
    expect(['b', 'A', 'item 10', 'item 9'].sort(compareAlphabetically)).toEqual(['A', 'b', 'item 9', 'item 10'])
  })
})

describe('sortByLabel', () => {
  it('orders A→Z with dot-prefixed labels on top and id as the tiebreak', () => {
    const labels = ['Tasks', '.. Hub', '. Core', 'app', 'Docs', 'app']
    const nodes = labels.map((label, i) => ({ id: `p${5 - i}`, label }))

    expect(sortByLabel(nodes).map(n => `${n.label}|${n.id}`)).toEqual([
      '. Core|p3',
      '.. Hub|p4',
      'app|p0',
      'app|p2',
      'Docs|p1',
      'Tasks|p5'
    ])
  })

  it('does not mutate its input', () => {
    const input = [{ id: 'b', label: 'B' }, { id: 'a', label: 'A' }]

    sortByLabel(input)
    expect(input.map(n => n.id)).toEqual(['b', 'a'])
  })
})

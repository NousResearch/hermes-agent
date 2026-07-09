import { describe, expect, it } from 'vitest'

import { buildGroups } from './list'

describe('buildGroups', () => {
  it('groups each user message with following assistant turns', () => {
    const groups = buildGroups(
      [
        '0:u1:user:1',
        '1:a1:assistant:2',
        '2:t1:tool:3',
        '3:u2:user:1',
        '4:a2:assistant:4'
      ].join('\n')
    )

    expect(groups).toEqual([
      { id: 'u1', indices: [0, 1, 2], kind: 'turn', weight: 6 },
      { id: 'u2', indices: [3, 4], kind: 'turn', weight: 5 }
    ])
  })
})

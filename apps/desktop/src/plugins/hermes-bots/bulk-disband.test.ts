/**
 * Pure helpers for bulk disband + multi-select on the Bots list.
 * Run:
 *   pnpm --filter @hermes/desktop exec vitest run src/plugins/hermes-bots/bulk-disband.test.ts
 */
import { describe, expect, it, vi } from 'vitest'

import { buildGroupDisbandTargets, disbandGroupChats, nextGroupSelection } from './bulk-disband'

describe('nextGroupSelection', () => {
  it('ctrl toggles', () => {
    const order = ['a', 'b', 'c']
    let s = nextGroupSelection(new Set(), order, 'a', { ctrlKey: true, shiftKey: false }, null)
    expect([...s.selected]).toEqual(['a'])
    s = nextGroupSelection(s.selected, order, 'b', { ctrlKey: true, shiftKey: false }, s.anchor)
    expect([...s.selected].sort()).toEqual(['a', 'b'])
    s = nextGroupSelection(s.selected, order, 'a', { ctrlKey: true, shiftKey: false }, s.anchor)
    expect([...s.selected]).toEqual(['b'])
  })

  it('meta toggles like ctrl (mac)', () => {
    const order = ['a', 'b']
    const s = nextGroupSelection(new Set(['a']), order, 'b', { ctrlKey: false, metaKey: true, shiftKey: false }, 'a')
    expect([...s.selected].sort()).toEqual(['a', 'b'])
  })

  it('shift ranges', () => {
    const order = ['a', 'b', 'c', 'd']
    const s = nextGroupSelection(new Set(['b']), order, 'd', { ctrlKey: false, shiftKey: true }, 'b')
    expect([...s.selected].sort()).toEqual(['b', 'c', 'd'])
  })

  it('plain click replaces', () => {
    const order = ['a', 'b']
    const s = nextGroupSelection(new Set(['a']), order, 'b', { ctrlKey: false, shiftKey: false }, 'a')
    expect([...s.selected]).toEqual(['b'])
  })
})

describe('disbandGroupChats', () => {
  it('calls disband once per name in order and skips dupes', async () => {
    const calls: Array<{ members: unknown[]; name: string }> = []
    const disband = vi.fn(async (name: string, members: unknown[]) => {
      calls.push({ name, members })
    })
    const res = await disbandGroupChats(
      [
        { name: 'Alpha', members: [{ name: 'x' } as never] },
        { name: 'Beta', members: [] },
        { name: 'Alpha', members: [{ name: 'y' } as never] },
        { name: '', members: [] }
      ],
      disband
    )
    expect(res.disbanded).toEqual(['Alpha', 'Beta'])
    expect(calls.map(c => c.name)).toEqual(['Alpha', 'Beta'])
    expect(calls).toHaveLength(2)
  })

  it('propagates disband failure after earlier successes', async () => {
    let n = 0
    await expect(
      disbandGroupChats(
        [
          { name: 'Ok', members: [] },
          { name: 'Bad', members: [] }
        ],
        async name => {
          n += 1
          if (name === 'Bad') {
            throw new Error('boom')
          }
        }
      )
    ).rejects.toThrow('boom')
    expect(n).toBe(2)
  })
})

describe('buildGroupDisbandTargets', () => {
  it('dedupes and resolves members', () => {
    const targets = buildGroupDisbandTargets(['A', 'B', 'A', ''], name => [{ name: `m-${name}` } as never])
    expect(targets).toEqual([
      { name: 'A', members: [{ name: 'm-A' }] },
      { name: 'B', members: [{ name: 'm-B' }] }
    ])
  })
})

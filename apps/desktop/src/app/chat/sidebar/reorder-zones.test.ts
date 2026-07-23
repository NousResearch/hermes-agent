import { describe, expect, it } from 'vitest'

import {
  reorderIds,
  type ReorderZoneSnapshot,
  resolveReorderTarget
} from './reorder-zones'

// A zone spanning x[0,100] y[0,100] with three rows at mid-Y 20/50/80.
function zone(ids: string[], mids: number[]): ReorderZoneSnapshot {
  return {
    bottom: 100,
    ids,
    left: 0,
    onReorder: () => undefined,
    right: 100,
    rows: ids.map((id, i) => ({ id, mid: mids[i]! })),
    top: 0
  }
}

describe('resolveReorderTarget', () => {
  const snap = [zone(['a', 'b', 'c'], [20, 50, 80])]

  it('returns null when the pointer is outside every zone', () => {
    expect(resolveReorderTarget(snap, 'a', 200, 50)).toBeNull()
    expect(resolveReorderTarget(snap, 'a', 50, -10)).toBeNull()
  })

  it('inserts before the first row whose mid-Y is below the pointer', () => {
    // y=10 is above row a's mid (20) → insert before a.
    expect(resolveReorderTarget(snap, 'c', 50, 10)).toEqual({
      before: 'a',
      ids: ['a', 'b', 'c'],
      onReorder: expect.any(Function)
    })
    // y=60 is below a(20) and b(50), above c(80) → insert before c.
    expect(resolveReorderTarget(snap, 'a', 50, 60)?.before).toBe('c')
  })

  it('resolves to end (before:null) below the last row', () => {
    expect(resolveReorderTarget(snap, 'a', 50, 95)?.before).toBeNull()
  })

  it('skips the dragged row when choosing the boundary', () => {
    // Dragging b, pointer at y=45 (just above b's own mid): b is skipped, so the
    // first OTHER row below the pointer is c → before c, not before b.
    expect(resolveReorderTarget(snap, 'b', 50, 45)?.before).toBe('c')
  })

  it('returns null when the zone does not own the dragged id (foreign drop)', () => {
    // A non-pinned row dragged over the pinned zone is not a reorder.
    expect(resolveReorderTarget(snap, 'x', 50, 50)).toBeNull()
  })

  it('picks the zone the pointer is actually inside among several', () => {
    const two: ReorderZoneSnapshot[] = [
      { ...zone(['a', 'b'], [20, 60]), top: 0, bottom: 80 },
      { ...zone(['p', 'q'], [120, 160]), top: 100, bottom: 180 }
    ]

    // Pointer in zone 2, dragging p: q is the first other row below → before q.
    expect(resolveReorderTarget(two, 'p', 50, 130)?.before).toBe('q')
    // Pointer near the top of zone 2, dragging q: p is below the pointer → before p.
    expect(resolveReorderTarget(two, 'q', 50, 110)?.before).toBe('p')
    // Pointer in zone 1, dragging b: a is below the pointer → before a.
    expect(resolveReorderTarget(two, 'b', 50, 10)?.before).toBe('a')
  })
})

describe('reorderIds', () => {
  it('moves an id to sit before the target', () => {
    expect(reorderIds(['a', 'b', 'c'], 'c', 'a')).toEqual(['c', 'a', 'b'])
    expect(reorderIds(['a', 'b', 'c'], 'a', 'c')).toEqual(['b', 'a', 'c'])
  })

  it('moves an id to the end when before is null', () => {
    expect(reorderIds(['a', 'b', 'c'], 'a', null)).toEqual(['b', 'c', 'a'])
  })

  it('returns the SAME reference on a no-op move (skip the commit)', () => {
    const ids = ['a', 'b', 'c']
    // Moving a before b is where a already is → unchanged.
    expect(reorderIds(ids, 'a', 'b')).toBe(ids)
    // Moving c to the end where it already is → unchanged.
    expect(reorderIds(ids, 'c', null)).toBe(ids)
  })

  it('returns the same reference when the id is not in the list', () => {
    const ids = ['a', 'b']
    expect(reorderIds(ids, 'z', 'a')).toBe(ids)
  })

  it('treats an unknown before-target as end insertion', () => {
    expect(reorderIds(['a', 'b', 'c'], 'a', 'zzz')).toEqual(['b', 'c', 'a'])
  })
})

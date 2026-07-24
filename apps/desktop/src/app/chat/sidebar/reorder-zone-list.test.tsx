import { act, cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ReorderZoneList } from './reorder-zone-list'
import { $sidebarReorderHint, snapshotReorderZones } from './reorder-zones'

afterEach(() => {
  cleanup()
  $sidebarReorderHint.set(null)
})

function items(ids: string[]) {
  return ids.map(id => ({ id, node: <div data-testid={`row-${id}`}>{id}</div> }))
}

describe('ReorderZoneList', () => {
  it('registers a reorder zone while mounted and unregisters on unmount', () => {
    const { unmount } = render(<ReorderZoneList items={items(['a', 'b', 'c'])} onReorder={() => undefined} />)

    // The drag resolver discovers the zone via snapshotReorderZones().
    const snaps = snapshotReorderZones()
    expect(snaps).toHaveLength(1)
    expect(snaps[0]!.ids).toEqual(['a', 'b', 'c'])

    unmount()
    expect(snapshotReorderZones()).toHaveLength(0)
  })

  it('tags each row so the resolver can find it by id', () => {
    render(<ReorderZoneList items={items(['a', 'b'])} onReorder={() => undefined} />)

    expect(document.querySelector('[data-reorder-row="a"]')).toBeTruthy()
    expect(document.querySelector('[data-reorder-row="b"]')).toBeTruthy()
  })

  it('exposes live ids to the registration (a reorder reads current order at engage)', () => {
    const onReorder = vi.fn()
    const { rerender } = render(<ReorderZoneList items={items(['a', 'b', 'c'])} onReorder={onReorder} />)

    // Re-render with a new order (as a commit would produce) — the zone's
    // getIds must report the latest, not the mount-time snapshot.
    rerender(<ReorderZoneList items={items(['c', 'a', 'b'])} onReorder={onReorder} />)

    expect(snapshotReorderZones()[0]!.ids).toEqual(['c', 'a', 'b'])
  })

  it('paints the insertion line only for a drag of one of its own rows', () => {
    render(<ReorderZoneList items={items(['a', 'b', 'c'])} onReorder={() => undefined} />)

    // A foreign drag (id not in this list) shows no line.
    act(() => $sidebarReorderHint.set({ before: 'b', draggedId: 'foreign' }))
    expect(document.querySelector('[data-reorder-line]')).toBeNull()

    // A drag of one of ours renders exactly one insertion line.
    act(() => $sidebarReorderHint.set({ before: 'b', draggedId: 'a' }))
    expect(document.querySelectorAll('[data-reorder-line]')).toHaveLength(1)
  })
})

import { describe, expect, it } from 'vitest'

import { KANBAN_LAYOUT } from './layout'

function classes(value: string): Set<string> {
  return new Set(value.split(/\s+/))
}

describe('Kanban responsive layout contract', () => {
  it('gives touch lanes one board width and disables snapping on desktop', () => {
    const rail = classes(KANBAN_LAYOUT.laneRail)
    const lane = classes(KANBAN_LAYOUT.lane)
    const collapsed = classes(KANBAN_LAYOUT.collapsedLane)

    expect(rail).toEqual(expect.objectContaining(new Set(['snap-x', 'snap-mandatory', 'md:snap-none'])))
    expect(lane).toEqual(expect.objectContaining(new Set(['w-full', 'snap-start', 'md:w-64', 'md:snap-none'])))
    expect(collapsed).toEqual(expect.objectContaining(new Set(['w-full', 'snap-start', 'md:w-8', 'md:snap-none'])))
  })

  it('contains the toolbar and drawer on narrow viewports while retaining desktop sizing', () => {
    const page = classes(KANBAN_LAYOUT.page)
    const filters = classes(KANBAN_LAYOUT.toolbarFilters)
    const drawer = classes(KANBAN_LAYOUT.drawer)

    expect(page).toContain('min-w-0')
    expect(filters).toEqual(expect.objectContaining(new Set(['w-full', 'min-w-0', 'md:w-auto'])))
    expect(drawer).toEqual(
      expect.objectContaining(
        new Set([
          'w-full',
          'max-w-full',
          'md:w-[26rem]',
          'pl-[env(safe-area-inset-left)]',
          'pr-[env(safe-area-inset-right)]',
          'pb-[env(safe-area-inset-bottom)]'
        ])
      )
    )
  })
})

import { describe, expect, it } from 'vitest'

import { mergePositionedNav, type PositionedNavItem } from './nav-order'

describe('mergePositionedNav', () => {
  const core: PositionedNavItem[] = [
    { id: 'new-session', route: '/' },
    { id: 'skills', route: '/skills' },
    { id: 'messaging', route: '/messaging' },
    { id: 'artifacts', route: '/artifacts' }
  ]

  it('places manifest nav relative to core rows', () => {
    const result = mergePositionedNav(core, [
      { id: 'kanban', position: 'after:skills' },
      { id: 'notes', position: 'before:artifacts' }
    ])

    expect(result.map(item => item.id)).toEqual(['new-session', 'skills', 'kanban', 'messaging', 'notes', 'artifacts'])
  })

  it('replaces a core row only when an explicit override target is declared', () => {
    const result = mergePositionedNav(
      [
        { id: 'new-session', route: '/' },
        { id: 'skills', route: '/skills' },
        { id: 'messaging', route: '/messaging' }
      ],
      [
        { id: 'plugin-home', override: '/', route: '/' },
        { id: 'kanban-skills', override: '/skills', route: '/skills' },
        { id: 'bad-collision', route: '/messaging' }
      ]
    )

    expect(result.map(item => item.id)).toEqual(['plugin-home', 'kanban-skills', 'messaging', 'bad-collision'])
  })

  it('preserves registry order and resolves contributed anchors', () => {
    const result = mergePositionedNav(core, [
      { id: 'board-a', position: 'after:skills' },
      { id: 'board-b', position: 'after:skills' },
      { id: 'board-child', position: 'after:board-b' }
    ])

    expect(result.map(item => item.id)).toEqual([
      'new-session',
      'skills',
      'board-a',
      'board-b',
      'board-child',
      'messaging',
      'artifacts'
    ])
  })

  it('appends unresolved and end positions stably', () => {
    const result = mergePositionedNav(core, [
      { id: 'unknown', position: 'after:missing' },
      { id: 'end', position: 'end' }
    ])

    expect(result.map(item => item.id)).toEqual(['new-session', 'skills', 'messaging', 'artifacts', 'end', 'unknown'])
  })
})

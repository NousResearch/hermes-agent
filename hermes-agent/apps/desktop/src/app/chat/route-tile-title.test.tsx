/**
 * A route tile opened BEFORE its plugin route registers must pick up the
 * contribution's title once registration lands. `paneMirror` recomputes titles
 * only when one of its atoms changes; without a routes-area signal the tab
 * kept the humanized-path fallback forever while the pane content healed.
 */
import { afterEach, beforeAll, expect, it } from 'vitest'

import { registry } from '@/contrib/registry'
import { $routeTiles, closeRouteTile, openRouteTile } from '@/store/route-tiles'

import { watchRouteTiles } from './route-tile'

const paneTitle = (path: string) => registry.getArea('panes').find(c => c.id === `route-tile:${path}`)?.title

beforeAll(() => {
  watchRouteTiles()
})

afterEach(() => {
  for (const tile of $routeTiles.get()) {
    closeRouteTile(tile.path)
  }
})

it('refreshes the tile title when its route registers after the tile opened', () => {
  openRouteTile('/late-atlas')
  expect(paneTitle('/late-atlas')).toBe('Late Atlas')

  const dispose = registry.register({
    area: 'routes',
    id: 'late-atlas:page',
    title: 'Atlas of Everything',
    data: { path: '/late-atlas' },
    render: () => null
  })

  expect(paneTitle('/late-atlas')).toBe('Atlas of Everything')
  dispose()
})

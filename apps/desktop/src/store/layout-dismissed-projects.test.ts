import { beforeEach, describe, expect, it } from 'vitest'

import { $dismissedAutoProjectIds, dismissAutoProject, filterVisibleProjects, restoreAutoProject } from './layout'

beforeEach(() => {
  $dismissedAutoProjectIds.set([])
})

const auto = (id: string) => ({ id, isAuto: true as const })
const explicit = (id: string) => ({ id, isAuto: false as const })
const home = { id: '__home__' }

describe('filterVisibleProjects', () => {
  it('drops dismissed autos and keeps explicit + home + undismissed', () => {
    const tree = [explicit('p_shop'), auto('/www/otl-theme'), auto('/www/keep'), home]
    expect(filterVisibleProjects(tree, ['/www/otl-theme', '/www/other']).map(p => p.id)).toEqual([
      'p_shop',
      '/www/keep',
      '__home__'
    ])
  })

  it('ignores dismiss ids that hit an explicit project', () => {
    // List is auto-only by construction; still don't let a stale id hide a real row.
    const tree = [explicit('p_real'), auto('/www/gone')]
    expect(filterVisibleProjects(tree, ['p_real', '/www/gone']).map(p => p.id)).toEqual(['p_real'])
  })

  it('restores a dismissed auto project (Undo round-trip) and is idempotent', () => {
    const tree = [auto('/www/hide-me'), auto('/www/stays')]
    dismissAutoProject('/www/hide-me')
    expect(filterVisibleProjects(tree, $dismissedAutoProjectIds.get()).map(p => p.id)).toEqual(['/www/stays'])

    restoreAutoProject('/www/hide-me')
    expect($dismissedAutoProjectIds.get()).toEqual([])
    expect(filterVisibleProjects(tree, $dismissedAutoProjectIds.get()).map(p => p.id)).toEqual([
      '/www/hide-me',
      '/www/stays'
    ])

    // Restoring an id that isn't dismissed must not throw or mutate the list.
    restoreAutoProject('/www/hide-me')
    expect($dismissedAutoProjectIds.get()).toEqual([])
  })
})

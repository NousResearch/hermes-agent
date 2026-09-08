import { describe, expect, it } from 'vitest'

import { filterVisibleProjects } from './layout'

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

  it('passes the list through when nothing is dismissed', () => {
    const tree = [auto('/www/a'), explicit('p_b')]
    expect(filterVisibleProjects(tree, [])).toBe(tree)
  })
})

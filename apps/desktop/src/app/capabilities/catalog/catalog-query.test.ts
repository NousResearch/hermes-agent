import { describe, expect, it } from 'vitest'

import type { CatalogEntry } from './catalog-data'
import {
  type CatalogFacets,
  catalogSortOptions,
  catalogSources,
  EMPTY_FACETS,
  filterCatalog,
  sortCatalog
} from './catalog-query'

const entry = (name: string, source: string) =>
  ({ id: name, name, source, category: 'tools', tags: [], search: name }) as unknown as CatalogEntry

const facets = (sources: string[]): CatalogFacets => ({
  sources,
  categories: [],
  tags: [],
  installedOnly: false,
  updatesOnly: false
})

const plugins = [
  entry('a', 'community'),
  entry('b', 'user'),
  entry('c', 'verified'),
  entry('d', 'bundled'),
  entry('e', 'official')
]

describe('plugin tier facet', () => {
  it('lists the tiers present in trust order and folds installed-only values in', () => {
    expect(catalogSources(plugins, 'plugins')).toEqual(['official', 'verified', 'community'])
    expect(catalogSources([entry('x', 'git'), entry('y', 'verified')], 'plugins')).toEqual(['verified', 'community'])
  })

  it('filters by the normalized tier', () => {
    const names = (sources: string[]) =>
      filterCatalog(plugins, facets(sources), '', () => false, 'plugins').map(row => row.name)

    expect(names(['official'])).toEqual(['d', 'e'])
    expect(names(['community'])).toEqual(['a', 'b'])
    expect(names(['verified', 'official'])).toEqual(['c', 'd', 'e'])
  })

  it('keeps skills on their raw registry source', () => {
    const skills = [entry('s1', 'clawhub'), entry('s2', 'official'), entry('s3', 'clawhub')]

    expect(catalogSources(skills, 'skills')).toEqual(['clawhub', 'official'])
    expect(filterCatalog(skills, facets(['clawhub']), '', () => false, 'skills').map(row => row.name)).toEqual([
      's1',
      's3'
    ])
  })
})

describe('installed-skill usage and updates', () => {
  const rows = [
    { ...entry('rare', 'local'), usage: 2 },
    { ...entry('feed', 'clawhub') },
    { ...entry('hot', 'hub'), usage: 40, updateAvailable: true },
    { ...entry('also', 'local'), usage: 2 }
  ] as CatalogEntry[]

  it('sorts by use count, most used first, ties A–Z, feed rows last', () => {
    expect(sortCatalog(rows, 'usage', 'skills').map(row => row.name)).toEqual(['hot', 'also', 'rare', 'feed'])
  })

  it('offers Most used only when something has been used', () => {
    const labels = { discover: 'd', usage: 'u', stars: 's', newest: 'n', updated: 'r', name: 'a' }

    expect(catalogSortOptions(rows, 'skills', labels).map(o => o.value)).toContain('usage')
    expect(catalogSortOptions([entry('x', 'local')], 'skills', labels).map(o => o.value)).not.toContain('usage')
  })

  it('narrows to entries with an update', () => {
    expect(
      filterCatalog(rows, { ...EMPTY_FACETS, updatesOnly: true }, '', () => true, 'skills').map(row => row.name)
    ).toEqual(['hot'])
  })
})

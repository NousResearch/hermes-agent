import { useState } from 'react'

import type { CatalogKind } from './catalog-data'
import { type CatalogFacets, type CatalogSort, EMPTY_FACETS } from './catalog-query'

type ListFacet = 'sources' | 'categories' | 'tags'

const toggled = (list: string[], value: string) =>
  list.includes(value) ? list.filter(item => item !== value) : [...list, value]

/** Rail state for one catalog. `onChange` runs after every change so the
 *  browser can drop paging and selection that no longer apply. */
export function useCatalogFilters(kind: CatalogKind, onChange: () => void) {
  const [facets, setFacets] = useState(EMPTY_FACETS)
  const [sort, setSortState] = useState<CatalogSort>(kind === 'plugins' ? 'stars' : 'discover')

  const update = (next: (current: CatalogFacets) => CatalogFacets) => {
    setFacets(next)
    onChange()
  }

  // `null` clears the facet (its "All" row).
  const toggle = (facet: ListFacet) => (value: string | null) =>
    update(current => ({ ...current, [facet]: value === null ? [] : toggled(current[facet], value) }))

  return {
    facets,
    sort,
    setSort: (value: CatalogSort) => {
      setSortState(value)
      onChange()
    },
    toggleSource: toggle('sources'),
    // A source whose descriptor was retired leaves the rail with its selection.
    dropSources: (ids: string[]) => update(current => ({ ...current, sources: current.sources.filter(id => !ids.includes(id)) })),
    toggleCategory: toggle('categories'),
    toggleTag: toggle('tags'),
    toggleInstalled: () => update(current => ({ ...current, installedOnly: !current.installedOnly })),
    toggleUpdates: () => update(current => ({ ...current, updatesOnly: !current.updatesOnly })),
    // Cards and "See all" drill into one category rather than toggling it.
    chooseCategory: (value: string) => update(current => ({ ...current, categories: [value], tags: [] })),
    clear: () => update(() => EMPTY_FACETS)
  }
}

/** Descriptor-driven sources come and go with their provider. A retired source
 *  leaves the rail together with its selection, so the list does not stay
 *  narrowed to a source that no longer exists. Derived during render from the
 *  previously offered set, so there is no effect and no extra frame. */
export function useSourceRetirement(sources: Record<string, string> | undefined, selected: string[], drop: (ids: string[]) => void) {
  const offered = Object.keys(sources ?? {})
  const offeredKey = offered.join('\n')
  const [knownKey, setKnownKey] = useState(offeredKey)

  if (knownKey === offeredKey) {
    return
  }

  setKnownKey(offeredKey)
  const retired = knownKey.split('\n').filter(id => id && !offered.includes(id) && selected.includes(id))

  if (retired.length) {
    drop(retired)
  }
}

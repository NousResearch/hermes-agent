import { useCallback, useMemo, useState } from 'react'

import type { CatalogKind } from './catalog-data'
import { type CatalogFacets, type CatalogSort, EMPTY_FACETS } from './catalog-query'

const toggled = (list: string[], value: string) =>
  list.includes(value) ? list.filter(item => item !== value) : [...list, value]

/** Rail state for one catalog. `onChange` runs after every change so the
 *  browser can drop paging and selection that no longer apply. */
export function useCatalogFilters(kind: CatalogKind, onChange: () => void) {
  const [facets, setFacets] = useState(EMPTY_FACETS)
  const [sort, setSortState] = useState<CatalogSort>(kind === 'plugins' ? 'stars' : 'discover')

  const update = useCallback(
    (next: (current: CatalogFacets) => CatalogFacets) => {
      setFacets(next)
      onChange()
    },
    [onChange]
  )

  // `null` clears the facet (its "All" row).
  const toggleSource = useCallback(
    (value: string | null) => update(current => ({ ...current, sources: value === null ? [] : toggled(current.sources, value) })),
    [update]
  )
  const toggleCategory = useCallback(
    (value: string | null) => update(current => ({ ...current, categories: value === null ? [] : toggled(current.categories, value) })),
    [update]
  )
  const toggleTag = useCallback(
    (value: string | null) => update(current => ({ ...current, tags: value === null ? [] : toggled(current.tags, value) })),
    [update]
  )
  const toggleInstalled = useCallback(
    () => update(current => ({ ...current, installedOnly: !current.installedOnly })),
    [update]
  )
  // Cards and "See all" drill into one category rather than toggling it.
  const chooseCategory = useCallback(
    (value: string) => update(current => ({ ...current, categories: [value], tags: [] })),
    [update]
  )
  const clear = useCallback(() => update(() => EMPTY_FACETS), [update])
  const setSort = useCallback(
    (value: CatalogSort) => {
      setSortState(value)
      onChange()
    },
    [onChange]
  )

  return useMemo(
    () => ({
      facets,
      sort,
      setSort,
      toggleSource,
      toggleCategory,
      toggleTag,
      toggleInstalled,
      chooseCategory,
      clear
    }),
    [facets, sort, setSort, toggleSource, toggleCategory, toggleTag, toggleInstalled, chooseCategory, clear]
  )
}

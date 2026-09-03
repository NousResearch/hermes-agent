import { beforeEach, describe, expect, it } from 'vitest'

import {
  $sidebarActiveFilterCount,
  $sidebarActiveFilterKinds,
  $sidebarCardRows,
  $sidebarFiltersActive,
  $sidebarGrouping,
  $sidebarOrdering,
  $sidebarPrFilter,
  $sidebarProfileFilter,
  $sidebarProjectFilter,
  $sidebarRowMeta,
  $sidebarShowArchived,
  $sidebarStatusFilter,
  $sidebarViewCustomized,
  clearSidebarFilters,
  resetSidebarView,
  setSidebarCardRows,
  setSidebarGrouping,
  setSidebarOrdering,
  sidebarActiveFilterCount,
  sidebarActiveFilterKinds,
  toggleSidebarPrFilter,
  toggleSidebarProfileFilter,
  toggleSidebarProjectFilter,
  toggleSidebarRowMeta,
  toggleSidebarStatusFilter
} from './layout'
import { $showAllProfiles } from './profile'

beforeEach(() => {
  $showAllProfiles.set(false)
  resetSidebarView()
})

describe('sidebarActiveFilterKinds / count', () => {
  it('names only the dimensions that are on, and counts each selected value', () => {
    expect(sidebarActiveFilterKinds([], [], [], [], false)).toEqual([])
    expect(sidebarActiveFilterCount([], [], [], [], false)).toBe(0)

    expect(sidebarActiveFilterKinds(['working', 'idle'], ['proj'], [], ['open'], true)).toEqual([
      'status',
      'project',
      'pr',
      'archived'
    ])
    expect(sidebarActiveFilterCount(['working', 'idle'], ['proj'], [], ['open'], true)).toBe(5)
  })
})

describe('clearSidebarFilters', () => {
  it('clears status, project, profile, pull request, and archived', () => {
    toggleSidebarStatusFilter('working')
    toggleSidebarProjectFilter('ceo-system')
    toggleSidebarProfileFilter('coder')
    toggleSidebarPrFilter('open')
    $sidebarShowArchived.set(true)

    expect($sidebarFiltersActive.get()).toBe(true)
    expect($sidebarActiveFilterCount.get()).toBe(5)
    expect($sidebarActiveFilterKinds.get()).toEqual(['status', 'project', 'profile', 'pr', 'archived'])

    clearSidebarFilters()

    expect($sidebarStatusFilter.get()).toEqual([])
    expect($sidebarProjectFilter.get()).toEqual([])
    expect($sidebarProfileFilter.get()).toEqual([])
    expect($sidebarPrFilter.get()).toEqual([])
    expect($sidebarShowArchived.get()).toBe(false)
    expect($sidebarFiltersActive.get()).toBe(false)
    expect($sidebarActiveFilterCount.get()).toBe(0)
    expect($sidebarActiveFilterKinds.get()).toEqual([])
  })

  it('leaves grouping, ordering, row metadata, and inbox style alone', () => {
    setSidebarGrouping('status')
    setSidebarOrdering('tokens')
    toggleSidebarRowMeta('tokens')
    setSidebarCardRows(true)
    toggleSidebarStatusFilter('unread')

    const grouping = $sidebarGrouping.get()
    const ordering = $sidebarOrdering.get()
    const rowMeta = $sidebarRowMeta.get()
    const cardRows = $sidebarCardRows.get()

    clearSidebarFilters()

    expect($sidebarGrouping.get()).toBe(grouping)
    expect($sidebarOrdering.get()).toBe(ordering)
    expect($sidebarRowMeta.get()).toEqual(rowMeta)
    expect($sidebarCardRows.get()).toBe(cardRows)
    expect($sidebarViewCustomized.get()).toBe(true)
    expect($sidebarFiltersActive.get()).toBe(false)
  })
})

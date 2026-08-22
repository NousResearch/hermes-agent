import { beforeEach, describe, expect, it } from 'vitest'

import {
  $sidebarGrouping,
  $sidebarOrdering,
  $sidebarRowMeta,
  $sidebarSessionOrderIds,
  $sidebarSessionOrderIdsByProfile,
  $sidebarViewCustomized,
  resetSidebarView,
  setSidebarGrouping,
  setSidebarOrdering,
  setSidebarSessionOrderIds,
  setSidebarSessionOrderIdsForProfile,
  toggleSidebarRowMeta,
  toggleSidebarStatusFilter
} from './layout'
import { $showAllProfiles } from './profile'

beforeEach(() => {
  $showAllProfiles.set(false)
  resetSidebarView()
})

describe('the sidebar as it ships', () => {
  it('groups by date, sorts by recency, and pins the timestamp and preview', () => {
    expect($sidebarGrouping.get()).toBe('date')
    expect($sidebarOrdering.get()).toBe('updated')
    expect($sidebarRowMeta.get()).toEqual(['preview', 'updated'])
  })

  it('offers no reset until something actually moves off the defaults', () => {
    expect($sidebarViewCustomized.get()).toBe(false)

    toggleSidebarRowMeta('tokens')

    expect($sidebarViewCustomized.get()).toBe(true)
  })

  it('is what reset puts back — every knob, not just the filters', () => {
    setSidebarGrouping('project')
    setSidebarOrdering('manual')
    setSidebarSessionOrderIdsForProfile('alpha', ['a2', 'a1'])
    setSidebarOrdering('cost')
    toggleSidebarRowMeta('updated')
    toggleSidebarRowMeta('cost')
    toggleSidebarStatusFilter('working')

    resetSidebarView()

    expect($sidebarGrouping.get()).toBe('date')
    expect($sidebarOrdering.get()).toBe('updated')
    expect($sidebarSessionOrderIdsByProfile.get()).toEqual({})
    expect($sidebarRowMeta.get()).toEqual(['preview', 'updated'])
    expect($sidebarViewCustomized.get()).toBe(false)
  })

  it('ships by date in the all-profiles scope too, and resets back to it', () => {
    $showAllProfiles.set(true)
    setSidebarGrouping('profile')

    resetSidebarView()

    expect($sidebarGrouping.get()).toBe('date')
    expect($sidebarViewCustomized.get()).toBe(false)
  })

  it('resets the scope the user is not looking at, so flipping the rail cannot restore it', () => {
    setSidebarGrouping('status')
    $showAllProfiles.set(true)
    setSidebarGrouping('profile')

    resetSidebarView()
    $showAllProfiles.set(false)

    expect($sidebarGrouping.get()).toBe('date')
  })

  it('turns all-profiles on when the user groups by profile, since that is the ask', () => {
    setSidebarGrouping('profile')

    expect($showAllProfiles.get()).toBe(true)
    expect($sidebarGrouping.get()).toBe('profile')
  })

  it('moves all-profiles Manual into profile grouping so the selected mode has reorder handles', () => {
    $showAllProfiles.set(true)
    setSidebarGrouping('status')

    setSidebarOrdering('manual')

    expect($sidebarOrdering.get()).toBe('manual')
    expect($sidebarGrouping.get()).toBe('profile')
  })

  it('moves scoped Project grouping back to the flat recents list when Manual is selected', () => {
    setSidebarGrouping('project')

    setSidebarOrdering('manual')

    expect($sidebarOrdering.get()).toBe('manual')
    expect($sidebarGrouping.get()).toBe('date')
  })

  it('persists colliding session ids independently per normalized profile', () => {
    setSidebarSessionOrderIdsForProfile(' alpha ', ['shared', 'alpha-2'])
    setSidebarSessionOrderIdsForProfile('beta', ['beta-2', 'shared'])

    expect($sidebarSessionOrderIdsByProfile.get()).toEqual({
      alpha: ['shared', 'alpha-2'],
      beta: ['beta-2', 'shared']
    })
  })

  it('restores every per profile manual order after using another ordering', () => {
    setSidebarOrdering('manual')
    setSidebarSessionOrderIds(['legacy-2', 'legacy-1'])
    setSidebarSessionOrderIdsForProfile('alpha', ['a2', 'a1'])
    setSidebarSessionOrderIdsForProfile('beta', ['b2', 'b1'])

    setSidebarOrdering('updated')
    expect($sidebarOrdering.get()).toBe('updated')
    expect($sidebarSessionOrderIds.get()).toEqual([])
    expect($sidebarSessionOrderIdsByProfile.get()).toEqual({
      alpha: ['a2', 'a1'],
      beta: ['b2', 'b1']
    })

    setSidebarOrdering('manual')
    expect($sidebarOrdering.get()).toBe('manual')
    expect($sidebarSessionOrderIds.get()).toEqual([])
    expect($sidebarSessionOrderIdsByProfile.get()).toEqual({
      alpha: ['a2', 'a1'],
      beta: ['b2', 'b1']
    })
  })
})

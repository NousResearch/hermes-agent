import { beforeEach, describe, expect, it } from 'vitest'

import {
  $sidebarGrouping,
  $sidebarOrdering,
  $sidebarRowMeta,
  $sidebarViewCustomized,
  resetSidebarView,
  setSidebarGrouping,
  setSidebarOrdering,
  toggleSidebarRowMeta,
  toggleSidebarStatusFilter
} from './layout'
import { $showAllProfiles } from './profile'

beforeEach(() => {
  $showAllProfiles.set(false)
  resetSidebarView()
})

describe('the sidebar as it ships', () => {
  it('groups by date, sorts by recency, and pins the timestamp', () => {
    expect($sidebarGrouping.get()).toBe('date')
    expect($sidebarOrdering.get()).toBe('updated')
    expect($sidebarRowMeta.get()).toEqual(['updated'])
  })

  it('offers no reset until something actually moves off the defaults', () => {
    expect($sidebarViewCustomized.get()).toBe(false)

    toggleSidebarRowMeta('tokens')

    expect($sidebarViewCustomized.get()).toBe(true)
  })

  it('is what reset puts back — every knob, not just the filters', () => {
    setSidebarGrouping('project')
    setSidebarOrdering('cost')
    toggleSidebarRowMeta('updated')
    toggleSidebarRowMeta('cost')
    toggleSidebarStatusFilter('working')

    resetSidebarView()

    expect($sidebarGrouping.get()).toBe('date')
    expect($sidebarOrdering.get()).toBe('updated')
    expect($sidebarRowMeta.get()).toEqual(['updated'])
    expect($sidebarViewCustomized.get()).toBe(false)
  })

  it('resets to the all-profiles default while that scope is on', () => {
    $showAllProfiles.set(true)
    setSidebarGrouping('status')

    resetSidebarView()

    expect($sidebarGrouping.get()).toBe('profile')
    expect($sidebarViewCustomized.get()).toBe(false)
  })

  it('turns all-profiles on when the user groups by profile, since that is the ask', () => {
    setSidebarGrouping('profile')

    expect($showAllProfiles.get()).toBe(true)
    expect($sidebarGrouping.get()).toBe('profile')
  })
})

import { beforeEach, describe, expect, it, vi } from 'vitest'

import { readKey } from '@/lib/storage'

import {
  $sidebarGrouping,
  $sidebarOrdering,
  $sidebarProjectDataWanted,
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
  it('groups by date, sorts by recency, and pins the timestamp and preview', () => {
    expect($sidebarGrouping.get()).toBe('date')
    expect($sidebarOrdering.get()).toBe('updated')
    expect($sidebarRowMeta.get()).toEqual(['preview', 'updated'])
  })

  // The persisted codec is an ALLOW-LIST: `listOf(ROW_META)` drops any id it
  // does not recognize while DECODING the stored record. Toggling the atom and
  // reading it back in this module never reaches that decode — the seed is read
  // exactly once, when the store module is first imported. So write the record
  // through the real toggle, then boot a fresh instance of the store the way a
  // reload would and require the option to come back intact. Drop `project`
  // from ROW_META and this is the assertion that goes red.
  it('reads the Project detail back from storage after a reload', async () => {
    toggleSidebarRowMeta('project')

    const saved = $sidebarRowMeta.get()

    expect(saved).toContain('project')
    expect(readKey('hermes.desktop.sidebarRowMeta')).toContain('project')

    vi.resetModules()

    const reloaded = await import('./layout')

    expect(reloaded.$sidebarRowMeta.get()).toEqual(saved)
  })

  // The Project detail NAMES a row off `$projects`, which only `projects.list`
  // fills — and the sidebar only issues that fetch for a view that asked for
  // project data. The flat, date-grouped list never does, so switching the
  // detail on has to raise the same kind of "someone needs this" signal the PR
  // badge raises, or the chip paints from whatever a past visit to the grouped
  // view happened to leave in the atom. Off by default: nobody who hasn't
  // asked pays for the round trip.
  it('asks for project data only once the Project detail is switched on', () => {
    expect($sidebarProjectDataWanted.get()).toBe(false)

    toggleSidebarRowMeta('project')

    expect($sidebarProjectDataWanted.get()).toBe(true)

    toggleSidebarRowMeta('project')

    expect($sidebarProjectDataWanted.get()).toBe(false)
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
})

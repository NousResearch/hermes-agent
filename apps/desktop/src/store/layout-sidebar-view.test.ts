import { beforeEach, describe, expect, it } from 'vitest'

import {
  $sidebarGrouping,
  $sidebarOrdering,
  $sidebarRecencyFilter,
  $sidebarRowMeta,
  $sidebarShowAllSessions,
  $sidebarViewCustomized,
  resetSidebarView,
  setSidebarGrouping,
  setSidebarOrdering,
  setSidebarShowAllSessions,
  toggleSidebarRowMeta,
  toggleSidebarRecencyFilter,
  toggleSidebarStatusFilter
} from './layout'
import { $showAllProfiles } from './profile'
import { makeSessionInfo } from '@/test/session-info'
import { sessionMatchesRecencyFilter } from '@/app/chat/sidebar/projects/workspace-groups'

beforeEach(() => {
  $showAllProfiles.set(false)
  resetSidebarView()
})

describe('the sidebar as it ships', () => {
  it('remembers expanded project previews across grouping changes and clears them on reset', () => {
    expect($sidebarShowAllSessions.get()).toBe(false)
    setSidebarGrouping('project')
    setSidebarShowAllSessions(true)
    setSidebarGrouping('date')

    expect($sidebarShowAllSessions.get()).toBe(true)
    expect($sidebarViewCustomized.get()).toBe(true)
    expect(window.localStorage.getItem('hermes.desktop.sidebarShowAllSessions')).toBe('true')

    resetSidebarView()

    expect($sidebarShowAllSessions.get()).toBe(false)
    expect($sidebarViewCustomized.get()).toBe(false)
    expect(window.localStorage.getItem('hermes.desktop.sidebarShowAllSessions')).toBe('false')
  })

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

  it('persists a recency window and clears it with the rest of the filters', () => {
    toggleSidebarRecencyFilter('1d')

    expect($sidebarRecencyFilter.get()).toEqual(['1d'])
    expect($sidebarViewCustomized.get()).toBe(true)

    resetSidebarView()

    expect($sidebarRecencyFilter.get()).toEqual([])
    expect($sidebarViewCustomized.get()).toBe(false)
  })

  it('includes boundary activity and falls back to started_at for recency windows', () => {
    const now = 2_000_000

    expect(sessionMatchesRecencyFilter(makeSessionInfo({ last_active: now - 86_400 }), ['1d'], now)).toBe(true)
    expect(sessionMatchesRecencyFilter(makeSessionInfo({ last_active: now - 86_401 }), ['1d'], now)).toBe(false)
    expect(
      sessionMatchesRecencyFilter(makeSessionInfo({ last_active: 0, started_at: now - 172_800 }), ['2d'], now)
    ).toBe(true)
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

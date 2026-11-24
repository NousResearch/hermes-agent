import { beforeEach, describe, expect, it } from 'vitest'

import {
  $sidebarAgentsGrouped,
  $sidebarGrouping,
  $sidebarOrdering,
  $sidebarRowMeta,
  $sidebarShowAllSessions,
  $sidebarViewCustomized,
  autoEnterProjectView,
  cycleSidebarGrouping,
  resetSidebarView,
  setSidebarAgentsGrouped,
  setSidebarGrouping,
  setSidebarOrdering,
  setSidebarShowAllSessions,
  SIDEBAR_GROUPING_ORDER,
  type SidebarGrouping,
  toggleSidebarRowMeta,
  toggleSidebarStatusFilter
} from './layout'
import { $showAllProfiles } from './profile'
import { $projectScope, ALL_PROJECTS, exitProjectScope } from './project-scope'

beforeEach(() => {
  $showAllProfiles.set(false)
  resetSidebarView()
  $projectScope.set(ALL_PROJECTS)
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

  it('visits every grouping once per lap and comes back to where it started', () => {
    const start = $sidebarGrouping.get()
    const visited: SidebarGrouping[] = []

    for (let step = 0; step < SIDEBAR_GROUPING_ORDER.length; step++) {
      cycleSidebarGrouping()
      visited.push($sidebarGrouping.get())

      if ($sidebarGrouping.get() === 'profile') {
        expect($showAllProfiles.get()).toBe(true)
      }
    }

    expect(new Set(visited)).toEqual(new Set(SIDEBAR_GROUPING_ORDER))
    expect($sidebarGrouping.get()).toBe(start)
  })
})

// An automatic follow (a session's cwd moving into a project) borrows the
// project tree for the duration of the detour. The stored grouping is the
// user's own choice and has to survive it.
describe('a project view borrowed by an automatic follow', () => {
  it('shows the project tree without writing the grouping preference', () => {
    setSidebarGrouping('date')
    window.localStorage.removeItem('hermes.desktop.agentsGroupedByWorkspace')

    autoEnterProjectView()

    expect($sidebarGrouping.get()).toBe('project')
    expect($sidebarAgentsGrouped.get()).toBe(false)
    expect(window.localStorage.getItem('hermes.desktop.agentsGroupedByWorkspace')).toBeNull()
    // A detour is not a customization: "Reset" should stay out of the menu.
    expect($sidebarViewCustomized.get()).toBe(false)
  })

  it('gives the sidebar back to the stored grouping when the scope is left', () => {
    setSidebarGrouping('date')
    $projectScope.set('p_x')
    autoEnterProjectView()

    exitProjectScope()

    expect($sidebarGrouping.get()).toBe('date')
    expect($projectScope.get()).toBe(ALL_PROJECTS)
  })

  it('keeps a project grouping the user picked themselves', () => {
    setSidebarGrouping('project')
    $projectScope.set('p_x')

    // The automatic detour must not stack on a project view the user chose.
    autoEnterProjectView()
    expect($sidebarGrouping.get()).toBe('project')

    exitProjectScope()

    expect($sidebarGrouping.get()).toBe('project')
    expect($sidebarAgentsGrouped.get()).toBe(true)
  })

  it('ends the detour as soon as the user picks a grouping', () => {
    autoEnterProjectView()

    setSidebarGrouping('status')

    expect($sidebarGrouping.get()).toBe('status')
  })

  it('an explicit un-group ends a detour that is already on screen', () => {
    setSidebarGrouping('date')
    autoEnterProjectView()

    // Any explicit write replaces the detour; it must not keep overriding the
    // grouping the setter just stored.
    setSidebarAgentsGrouped(false)

    expect($sidebarGrouping.get()).toBe('date')
  })

  it('restores the all-profiles grouping it was borrowed from', () => {
    setSidebarGrouping('profile')

    autoEnterProjectView()
    expect($sidebarGrouping.get()).toBe('project')

    exitProjectScope()
    expect($sidebarGrouping.get()).toBe('profile')
  })
})

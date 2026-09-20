import { expect, it } from 'vitest'

import { $sidebarFiltersActive, $sidebarTagFilter, resetSidebarView, toggleSidebarTagFilter } from './layout'
import { $sessions } from './session'
it('reset clears selected tag filters without touching assignments', () => {
  const sessions = $sessions.get()
  toggleSidebarTagFilter('one')
  toggleSidebarTagFilter('two')
  expect($sidebarTagFilter.get()).toEqual(['one', 'two'])
  expect($sidebarFiltersActive.get()).toBe(true)
  resetSidebarView()
  expect($sidebarTagFilter.get()).toEqual([])
  expect($sessions.get()).toBe(sessions)
})

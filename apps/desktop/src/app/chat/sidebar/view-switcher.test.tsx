import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $sidebarGrouping,
  setSidebarGrouping
} from '@/store/layout'
import { $showAllProfiles, setShowAllProfiles } from '@/store/profile'
import { $projectScope, ALL_PROJECTS, exitProjectScope } from '@/store/projects'

import { resolveSidebarView, SidebarViewSwitcher } from './view-switcher'

afterEach(cleanup)

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      sidebar: {
        viewAria: 'Session view',
        viewProfiles: 'Profiles',
        viewProjects: 'Projects',
        viewSessions: 'Sessions'
      }
    }
  })
}))

beforeEach(() => {
  // Reset to the shipped defaults so each test starts from the Sessions view.
  setSidebarGrouping('date')
  setShowAllProfiles(false)
  exitProjectScope()
})

describe('resolveSidebarView', () => {
  it('maps flat groupings to the Sessions view', () => {
    expect(resolveSidebarView('date', false)).toBe('sessions')
    expect(resolveSidebarView('status', false)).toBe('sessions')
  })

  it('maps workspace grouping to the Projects view', () => {
    expect(resolveSidebarView('project', false)).toBe('projects')
  })

  it('maps the all-profiles scope to the Profiles view', () => {
    expect(resolveSidebarView('project', true)).toBe('profiles')
    expect(resolveSidebarView('date', true)).toBe('profiles')
    expect(resolveSidebarView('profile', true)).toBe('profiles')
  })

  it('maps the all-profiles grouping to the Profiles view', () => {
    expect(resolveSidebarView('profile', false)).toBe('profiles')
  })
})

describe('SidebarViewSwitcher', () => {
  it('highlights the view the store state is in', () => {
    setSidebarGrouping('project')
    render(<SidebarViewSwitcher multiProfile />)

    expect(screen.getByRole('button', { name: 'Projects' }).getAttribute('aria-pressed')).toBe('true')
    expect(screen.getByRole('button', { name: 'Sessions' }).getAttribute('aria-pressed')).toBe('false')
  })

  it('omits the Profiles segment for single-profile users', () => {
    render(<SidebarViewSwitcher multiProfile={false} />)

    expect(screen.queryByRole('button', { name: 'Profiles' })).toBeNull()
    expect(screen.getByRole('button', { name: 'Sessions' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Projects' })).toBeTruthy()
  })

  it('switching to Projects sets workspace grouping and leaves all-profiles', () => {
    setShowAllProfiles(true)
    render(<SidebarViewSwitcher multiProfile />)

    fireEvent.click(screen.getByRole('button', { name: 'Projects' }))

    expect($showAllProfiles.get()).toBe(false)
    expect($sidebarGrouping.get()).toBe('project')
  })

  it('switching to Sessions leaves a project drill-in and sets date grouping', () => {
    $projectScope.set('p_123')
    render(<SidebarViewSwitcher multiProfile />)

    fireEvent.click(screen.getByRole('button', { name: 'Sessions' }))

    expect($projectScope.get()).toBe(ALL_PROJECTS)
    expect($sidebarGrouping.get()).toBe('date')
  })

  it('switching to Profiles enables the all-profiles scope grouped by owner', () => {
    render(<SidebarViewSwitcher multiProfile />)

    fireEvent.click(screen.getByRole('button', { name: 'Profiles' }))

    expect($showAllProfiles.get()).toBe(true)
    expect($sidebarGrouping.get()).toBe('profile')
  })
})

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'
import {
  $sidebarFiltersActive,
  $sidebarProjectFilter,
  $sidebarStatusFilter,
  clearSidebarFilters,
  resetSidebarView,
  toggleSidebarProjectFilter,
  toggleSidebarStatusFilter
} from '@/store/layout'
import { $showAllProfiles } from '@/store/profile'

import { SidebarFilterEmptyState } from './section-states'

afterEach(cleanup)

beforeEach(() => {
  $showAllProfiles.set(false)
  resetSidebarView()
})

function renderEmpty() {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <SidebarFilterEmptyState />
    </I18nProvider>
  )
}

describe('SidebarFilterEmptyState', () => {
  it('names the active filters and clears only those when clicked', () => {
    toggleSidebarStatusFilter('working')
    toggleSidebarProjectFilter('ceo-system')

    expect($sidebarFiltersActive.get()).toBe(true)

    renderEmpty()

    expect(screen.getByText('No sessions match the active filters.')).toBeTruthy()
    expect(screen.getByText('Active: Status · Project')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Clear filters' }))

    expect($sidebarStatusFilter.get()).toEqual([])
    expect($sidebarProjectFilter.get()).toEqual([])
    expect($sidebarFiltersActive.get()).toBe(false)
  })

  it('uses the exported clear action, not a full view reset', () => {
    toggleSidebarStatusFilter('idle')
    const before = $sidebarStatusFilter.get()

    expect(before).toEqual(['idle'])
    clearSidebarFilters()
    expect($sidebarStatusFilter.get()).toEqual([])
  })
})

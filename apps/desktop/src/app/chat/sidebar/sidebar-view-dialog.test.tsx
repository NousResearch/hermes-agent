import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { resetSidebarView, setSidebarGrouping } from '@/store/layout'
import { $showAllProfiles } from '@/store/profile'
import { $savedSidebarViews, saveCurrentSidebarView } from '@/store/sidebar-views'

import { SidebarViewDialog } from './sidebar-view-dialog'

beforeEach(() => {
  $showAllProfiles.set(false)
  resetSidebarView()
  $savedSidebarViews.set({ version: 1, views: [] })
})

afterEach(cleanup)

describe('SidebarViewDialog', () => {
  it('confirms replacing a saved view with the current sidebar configuration', async () => {
    const view = saveCurrentSidebarView('Review queue', { id: 'review', now: 100 })
    expect(view).not.toBeNull()

    setSidebarGrouping('none')

    render(<SidebarViewDialog dialog={{ kind: 'update', view: view! }} onClose={() => undefined} />)

    expect(await screen.findByRole('dialog', { name: 'Update saved view?' })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Update' }))

    expect($savedSidebarViews.get().views[0]).toMatchObject({
      createdAt: 100,
      id: 'review',
      name: 'Review queue',
      state: { grouping: 'none' }
    })
  })
})

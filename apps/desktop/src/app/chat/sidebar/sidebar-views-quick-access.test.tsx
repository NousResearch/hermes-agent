import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $sidebarGrouping, resetSidebarView, setSidebarGrouping } from '@/store/layout'
import { $showAllProfiles, setShowAllProfiles } from '@/store/profile'
import { $savedSidebarViews, saveCurrentSidebarView } from '@/store/sidebar-views'

import { SidebarSavedViewsQuickAccess } from './sidebar-views-quick-access'

class TestResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

vi.stubGlobal('ResizeObserver', TestResizeObserver)

beforeEach(() => {
  $showAllProfiles.set(false)
  resetSidebarView()
  $savedSidebarViews.set({ version: 1, views: [] })
})

afterEach(cleanup)

describe('SidebarSavedViewsQuickAccess', () => {
  it('stays out of the header until a view has been saved', () => {
    render(<SidebarSavedViewsQuickAccess />)

    expect(screen.queryByRole('button', { name: 'Saved views' })).toBeNull()
  })

  it('opens on hover and applies a saved view with one click', async () => {
    setSidebarGrouping('none')
    saveCurrentSidebarView('Overview', { id: 'overview', now: 100 })
    setSidebarGrouping('date')
    render(
      <>
        <input aria-label="Composer" />
        <SidebarSavedViewsQuickAccess />
      </>
    )

    const composer = screen.getByRole('textbox', { name: 'Composer' })
    composer.focus()
    fireEvent.pointerEnter(screen.getByRole('button', { name: 'Saved views' }))
    const item = await screen.findByRole('menuitem', { name: 'Overview' })

    await waitFor(() => expect(composer.matches(':focus')).toBe(true))
    fireEvent.click(item)

    expect($sidebarGrouping.get()).toBe('none')
  })

  it('opens as a keyboard menu and applies the focused view with Enter', async () => {
    setSidebarGrouping('none')
    saveCurrentSidebarView('Overview', { id: 'overview', now: 100 })
    setSidebarGrouping('date')
    render(<SidebarSavedViewsQuickAccess />)

    const trigger = screen.getByRole('button', { name: 'Saved views' })
    trigger.focus()
    fireEvent.keyDown(trigger, { key: 'ArrowDown' })

    const item = await screen.findByRole('menuitem', { name: 'Overview' })
    expect(item.matches(':focus')).toBe(true)
    fireEvent.keyDown(item, { key: 'Enter' })

    expect($sidebarGrouping.get()).toBe('none')
  })

  it('confirms before a saved view switches away from the current profile scope', async () => {
    setSidebarGrouping('none')
    saveCurrentSidebarView('Default overview', { id: 'default-overview', now: 100 })
    setSidebarGrouping('date')
    setShowAllProfiles(true)
    render(<SidebarSavedViewsQuickAccess />)

    fireEvent.pointerEnter(screen.getByRole('button', { name: 'Saved views' }))
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Default overview' }))

    expect(await screen.findByRole('dialog', { name: 'Switch profile and use view?' })).toBeTruthy()
    expect($sidebarGrouping.get()).toBe('date')

    fireEvent.click(screen.getByRole('button', { name: 'Switch and use view' }))

    expect($showAllProfiles.get()).toBe(false)
    expect($sidebarGrouping.get()).toBe('none')
  })
})

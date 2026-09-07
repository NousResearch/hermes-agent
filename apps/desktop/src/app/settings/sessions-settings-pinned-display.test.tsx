// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import {
  $sidebarPinnedCardRows,
  $sidebarPinnedInProjects,
  setSidebarPinnedCardRows,
  setSidebarPinnedInProjects
} from '@/store/layout'

import { PinnedDisplaySettings } from './sessions-settings'

beforeEach(() => {
  setSidebarPinnedCardRows(false)
  setSidebarPinnedInProjects(false)
})

afterEach(cleanup)

describe('PinnedDisplaySettings', () => {
  it('renders both opt-in toggles off by default', () => {
    render(<PinnedDisplaySettings />)

    expect(screen.getByRole('switch', { name: 'Display pinned threads in Inbox style' })).toBeTruthy()
    expect(screen.getByRole('switch', { name: 'Show pinned threads in their project / worktree' })).toBeTruthy()
    expect($sidebarPinnedCardRows.get()).toBe(false)
    expect($sidebarPinnedInProjects.get()).toBe(false)
  })

  it('Inbox style toggle flips the pinned card-rows atom', () => {
    render(<PinnedDisplaySettings />)

    fireEvent.click(screen.getByRole('switch', { name: 'Display pinned threads in Inbox style' }))
    expect($sidebarPinnedCardRows.get()).toBe(true)

    fireEvent.click(screen.getByRole('switch', { name: 'Display pinned threads in Inbox style' }))
    expect($sidebarPinnedCardRows.get()).toBe(false)
  })

  it('project / worktree toggle flips the pinned-in-projects atom', () => {
    render(<PinnedDisplaySettings />)

    fireEvent.click(screen.getByRole('switch', { name: 'Show pinned threads in their project / worktree' }))
    expect($sidebarPinnedInProjects.get()).toBe(true)

    fireEvent.click(screen.getByRole('switch', { name: 'Show pinned threads in their project / worktree' }))
    expect($sidebarPinnedInProjects.get()).toBe(false)
  })
})

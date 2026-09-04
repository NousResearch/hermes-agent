import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { SidebarViewSwitcher } from './view-switcher'

afterEach(cleanup)

describe('SidebarViewSwitcher', () => {
  it('marks the active view and exposes both explicit destinations', () => {
    render(
      <SidebarViewSwitcher
        active="sessions"
        ariaLabel="Sidebar view"
        onChange={vi.fn()}
        projectsLabel="Projects"
        sessionsLabel="Sessions"
      />
    )

    expect(screen.getByRole('button', { name: 'Sessions' }).getAttribute('aria-pressed')).toBe('true')
    expect(screen.getByRole('button', { name: 'Projects' }).getAttribute('aria-pressed')).toBe('false')
  })

  it('requests the selected view without opening a session', () => {
    const onChange = vi.fn()

    render(
      <SidebarViewSwitcher
        active="sessions"
        ariaLabel="Sidebar view"
        onChange={onChange}
        projectsLabel="Projects"
        sessionsLabel="Sessions"
      />
    )

    fireEvent.click(screen.getByRole('button', { name: 'Projects' }))
    expect(onChange).toHaveBeenCalledOnce()
    expect(onChange).toHaveBeenCalledWith('projects')
  })
})

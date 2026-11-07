import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { SidebarPanelLabel } from '@/app/shell/sidebar-label'

import { SidebarDateDivider } from './chrome'
import { WorkspaceShowMoreRow } from './projects/workspace-header'
import { SidebarPinnedEmptyState } from './section-states'

afterEach(cleanup)

// Themes and plugins style the sidebar's scaffolding by these slots, not by the
// Tailwind classes that paint it (plugin SDK: "the contract is the attribute").
describe('sidebar styling hooks', () => {
  it('names the section label and its marker', () => {
    const { container } = render(<SidebarPanelLabel>Pinned</SidebarPanelLabel>)

    expect(container.querySelector('[data-slot="sidebar-panel-label"]')?.textContent).toBe('Pinned')
    expect(container.querySelector('[data-slot="sidebar-panel-label-dot"]')).toBeTruthy()
  })

  it('names the date divider', () => {
    const { container } = render(<SidebarDateDivider label="Yesterday" />)

    expect(container.querySelector('[data-slot="sidebar-date-divider"]')?.textContent).toContain('Yesterday')
  })

  it('names the show-more row', () => {
    const { container } = render(<WorkspaceShowMoreRow label="Show all 18 sessions" onClick={() => undefined} />)

    expect(container.querySelector('[data-slot="sidebar-show-more"]')?.textContent).toContain('Show all 18 sessions')
  })

  it('tells the no-pins hint apart from a filter hiding every pin', () => {
    const { container, rerender } = render(<SidebarPinnedEmptyState archived={false} filtered={false} />)
    const empty = () => container.querySelector('[data-slot="sidebar-pinned-empty"]')

    expect(empty()).toBeTruthy()
    expect(empty()?.hasAttribute('data-filtered')).toBe(false)

    rerender(<SidebarPinnedEmptyState archived={false} filtered />)

    expect(empty()?.hasAttribute('data-filtered')).toBe(true)
  })
})

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { SidebarDateDivider } from './chrome'

afterEach(cleanup)

describe('SidebarDateDivider', () => {
  it('collapses the group when the caption is clicked', () => {
    const onToggle = vi.fn()

    render(
      <SidebarDateDivider
        label="Yesterday"
        toggle={{ ariaLabel: 'Hide Yesterday sessions', onToggle, open: true }}
      />
    )

    fireEvent.click(screen.getByRole('button', { name: 'Hide Yesterday sessions' }))
    expect(onToggle).toHaveBeenCalledOnce()
    expect(screen.getByRole('button', { name: 'Hide Yesterday sessions' }).getAttribute('aria-expanded')).toBe('true')
  })

  it('stays a static caption when it is not collapsible', () => {
    render(<SidebarDateDivider label="Yesterday" />)

    expect(screen.queryByRole('button')).toBeNull()
    expect(screen.getByText('Yesterday')).toBeTruthy()
  })
})

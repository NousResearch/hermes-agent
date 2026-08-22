import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { SidebarDateDivider } from './chrome'

describe('SidebarDateDivider', () => {
  it('exposes a native disclosure button with the visible label and expanded state', () => {
    const onToggle = vi.fn()

    const { rerender } = render(
      <SidebarDateDivider label="Today" toggle={{ ariaLabel: 'Collapse Today', onToggle, open: true }} />
    )

    const button = screen.getByRole('button', { name: 'Collapse Today' })

    expect(button.getAttribute('aria-expanded')).toBe('true')
    expect(button.textContent).toContain('Today')

    fireEvent.click(button)
    expect(onToggle).toHaveBeenCalledTimes(1)

    rerender(<SidebarDateDivider label="Today" toggle={{ ariaLabel: 'Expand Today', onToggle, open: false }} />)

    expect(screen.getByRole('button', { name: 'Expand Today' }).getAttribute('aria-expanded')).toBe('false')
  })
})

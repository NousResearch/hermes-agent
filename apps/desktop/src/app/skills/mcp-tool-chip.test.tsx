// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { McpToolChip } from './mcp-tool-chip'

afterEach(() => cleanup())

describe('McpToolChip', () => {
  it('exposes server-reported details when the chip receives keyboard focus', async () => {
    render(
      <McpToolChip
        action="Disable read_page"
        details={'Read a page\nServer reports read-only'}
        enabled
        onToggle={vi.fn()}
        saved
        toolName="read_page"
      />
    )

    const chip = screen.getByRole('button', { name: 'Disable read_page' })
    vi.spyOn(chip, 'matches').mockImplementation(selector => selector === ':focus-visible')

    fireEvent.keyDown(chip, { key: 'Tab' })
    fireEvent.focus(chip)

    await waitFor(() => {
      const tooltip = screen.getByRole('tooltip')

      expect(tooltip.textContent).toContain('Read a page')
      expect(tooltip.textContent).toContain('Server reports read-only')
    })
    expect(chip.hasAttribute('title')).toBe(false)
  })

  it('retains the existing toggle semantics', () => {
    const onToggle = vi.fn()
    render(<McpToolChip action="Disable read_page" details="" enabled onToggle={onToggle} saved toolName="read_page" />)

    fireEvent.click(screen.getByRole('button', { name: 'Disable read_page' }))

    expect(onToggle).toHaveBeenCalledOnce()
  })
})

// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { CommandsTab } from './commands-tab'

describe('CommandsTab', () => {
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('filters commands, copies the canonical token, and avoids raw argument tooltips', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } })
    render(<CommandsTab />)

    const branch = screen.getByRole('button', { name: /Copy \/branch/i })
    expect(branch.getAttribute('title')).toBeNull()

    fireEvent.click(branch)
    await waitFor(() => expect(writeText).toHaveBeenCalledWith('/branch'))

    fireEvent.change(screen.getByRole('searchbox', { name: 'Filter slash commands' }), {
      target: { value: 'branch' }
    })
    expect(screen.getByRole('button', { name: /Copy \/branch/i })).toBeTruthy()
    expect(screen.queryByRole('button', { name: /Copy \/model/i })).toBeNull()
  })
})

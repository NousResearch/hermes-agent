import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { Dialog, DialogContent, DialogTitle } from './dialog'

afterEach(cleanup)

describe('DialogContent close button', () => {
  it('closes the dialog when clicked', () => {
    const onOpenChange = vi.fn()
    render(
      <Dialog onOpenChange={onOpenChange} open>
        <DialogContent>
          <DialogTitle>Test dialog</DialogTitle>
        </DialogContent>
      </Dialog>
    )

    fireEvent.click(screen.getByRole('button', { name: /close/i }))
    expect(onOpenChange).toHaveBeenCalledWith(false)
  })

  it('does not show the tooltip immediately on open (no hover/focus yet)', async () => {
    render(
      <Dialog open>
        <DialogContent>
          <DialogTitle>Test dialog</DialogTitle>
        </DialogContent>
      </Dialog>
    )

    // Radix would otherwise autofocus the close button on open, which also
    // triggers the tooltip via focus. The dialog renders synchronously, so
    // no need to wait for the button — just assert no tooltip mounted.
    expect(screen.getByRole('button')).toBeTruthy()
    expect(screen.queryByRole('tooltip')).toBeNull()
  })

  it('shows the tooltip on focus (Radix opens on focus as well as hover; jsdom cannot reliably simulate real pointer hover)', async () => {
    render(
      <Dialog open>
        <DialogContent>
          <DialogTitle>Test dialog</DialogTitle>
        </DialogContent>
      </Dialog>
    )

    const closeButton = screen.getByRole('button', { name: /close/i })
    fireEvent.focus(closeButton)

    await waitFor(() => {
      const tooltip = screen.getByRole('tooltip')
      expect(tooltip.textContent).toMatch(/close/i)
    })
  })
})

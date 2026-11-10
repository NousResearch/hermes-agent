import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { Dialog, DialogContent, DialogTitle } from './dialog'
import { Sheet, SheetContent, SheetTitle } from './sheet'

afterEach(cleanup)

describe('DialogContent close button', () => {
  it('keeps initial focus on the form rather than the visually first close row', () => {
    render(
      <Dialog open>
        <DialogContent>
          <DialogTitle>Form</DialogTitle>
          <input aria-label="Name" />
        </DialogContent>
      </Dialog>
    )
    expect(globalThis.document.activeElement).toBe(screen.getByRole('textbox', { name: 'Name' }))
  })

  it('routes the shared sheet close through Radix', () => {
    const onOpenChange = vi.fn()
    render(
      <Sheet onOpenChange={onOpenChange} open>
        <SheetContent>
          <SheetTitle>Panel</SheetTitle>
        </SheetContent>
      </Sheet>
    )
    fireEvent.click(screen.getByRole('button', { name: /close/i }))
    expect(onOpenChange).toHaveBeenCalledWith(false)
  })

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
})

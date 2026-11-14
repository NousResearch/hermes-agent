import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { useState } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { Dialog, DialogContent, DialogTitle } from './dialog'

afterEach(cleanup)

function PageDialog({ banner, preventReturn = false, autoFocus = false }: { banner?: string; preventReturn?: boolean; autoFocus?: boolean }) {
  const [open, setOpen] = useState(false)
  return (
    <div data-page-surface>
      <button onClick={() => setOpen(true)}>Edit settings</button>
      <button onClick={() => setOpen(true)}>Edit another setting</button>
      <Dialog onOpenChange={setOpen} open={open}>
        <DialogContent banner={banner} onCloseAutoFocus={preventReturn ? event => event.preventDefault() : undefined}>
          <DialogTitle>Edit settings</DialogTitle>
          <input aria-label="Name" autoFocus={autoFocus} />
        </DialogContent>
      </Dialog>
    </div>
  )
}

it.each([
  [undefined, false],
  ['Save failed', false],
  [undefined, true],
  ['Save failed', true]
] as const)('restores a programmatic page dialog opener with banner %s and autofocus %s', async (banner, autoFocus) => {
  render(<PageDialog autoFocus={autoFocus} banner={banner} />)
  for (const name of ['Edit settings', 'Edit another setting']) {
    const opener = screen.getByRole('button', { name })
    opener.focus()
    fireEvent.click(opener)
    expect(document.activeElement).toBe(screen.getByRole('textbox'))
    fireEvent.keyDown(document.activeElement!, { key: 'Escape' })
    await vi.waitFor(() => expect(document.activeElement).toBe(opener))
  }
})

it('respects a dialog caller that owns close focus', async () => {
  render(<PageDialog preventReturn />)
  const opener = screen.getByRole('button', { name: 'Edit settings' })
  opener.focus()
  fireEvent.click(opener)
  fireEvent.keyDown(document.activeElement!, { key: 'Escape' })
  await vi.waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
  expect(document.activeElement).not.toBe(opener)
})

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
})

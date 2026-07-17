import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { Dialog, DialogContent, DialogTitle } from './dialog'

describe('DialogContent', () => {
  afterEach(() => {
    cleanup()
  })

  it('preserves close behavior when the close button has a tooltip', async () => {
    const onOpenChange = vi.fn()

    render(
      <Dialog onOpenChange={onOpenChange} open>
        <DialogContent>
          <DialogTitle>New profile</DialogTitle>
        </DialogContent>
      </Dialog>
    )

    const closeButton = screen.getByRole('button', { name: 'Close' })

    fireEvent.pointerMove(closeButton, { pointerType: 'mouse' })
    expect((await screen.findByRole('tooltip')).textContent).toContain('Close')

    fireEvent.click(closeButton)
    expect(onOpenChange).toHaveBeenCalledWith(false)
  })
})

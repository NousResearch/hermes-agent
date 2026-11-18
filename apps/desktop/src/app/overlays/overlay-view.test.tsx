import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { Dialog, DialogContent, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { $shellChoice } from '@/store/shell-variant'

import { OverlayView } from './overlay-view'

afterEach(() => {
  cleanup()
  $shellChoice.set(null)
})

it('keeps full-page destinations open on background clicks and closes explicitly', () => {
  $shellChoice.set(true)
  const close = vi.fn()
  const { container } = render(<OverlayView onClose={close}>Settings</OverlayView>)
  const surface = container.querySelector('[data-overlay-surface]')!

  fireEvent.click(surface)
  expect(close).not.toHaveBeenCalled()
  expect(screen.queryByRole('button', { name: 'Close' })).toBeNull()
  fireEvent.click(screen.getByRole('button', { name: 'Back' }))
  expect(close).toHaveBeenCalledOnce()
  close.mockClear()
  fireEvent.keyDown(window, { key: 'Escape' })
  expect(close).toHaveBeenCalledOnce()
})

it('restores backdrop dismissal when switching back to the classic layout', () => {
  $shellChoice.set(true)
  const close = vi.fn()
  const { container } = render(<OverlayView onClose={close}>Settings</OverlayView>)

  act(() => $shellChoice.set(false))
  expect(screen.queryByRole('button', { name: 'Back' })).toBeNull()
  expect(screen.getByRole('button', { name: 'Close' })).toBeTruthy()
  fireEvent.click(container.querySelector('[data-overlay-surface]')!)
  expect(close).toHaveBeenCalledOnce()
})

it('dismisses a nested dialog before the page and restores its trigger focus', async () => {
  $shellChoice.set(true)
  const close = vi.fn()
  render(
    <OverlayView onClose={close}>
      <Dialog>
        <DialogTrigger>Confirm action</DialogTrigger>
        <DialogContent aria-describedby={undefined}>
          <DialogTitle>Confirmation</DialogTitle>
          <button>Continue</button>
        </DialogContent>
      </Dialog>
    </OverlayView>
  )
  const trigger = screen.getByRole('button', { name: 'Confirm action' })
  trigger.focus()
  fireEvent.click(trigger)
  expect(screen.getByRole('dialog')).toBeTruthy()
  fireEvent.keyDown(document.activeElement!, { key: 'Escape' })
  expect(screen.queryByRole('dialog')).toBeNull()
  expect(close).not.toHaveBeenCalled()
  await vi.waitFor(() => expect(document.activeElement).toBe(trigger))
  fireEvent.keyDown(trigger, { key: 'Escape' })
  expect(close).toHaveBeenCalledOnce()
})

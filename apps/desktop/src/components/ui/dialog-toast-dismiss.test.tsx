import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { NotificationStack } from '@/components/notifications'
import { clearNotifications, notify } from '@/store/notifications'
import { stubResizeObserver } from '@/test/jsdom'

import { Dialog, DialogContent, DialogTitle } from './dialog'

beforeAll(stubResizeObserver)

// DismissableLayer registers its document listeners in a setTimeout(0), and
// resolves the deferred dismissal through another one.
const flushTimers = () => new Promise(resolve => setTimeout(resolve, 10))

// A real click: pointer sequence, focus moving to the pressed button, click.
function press(element: HTMLElement) {
  fireEvent.pointerDown(element, { button: 0 })
  fireEvent.mouseDown(element, { button: 0 })
  act(() => element.focus())
  fireEvent.pointerUp(element, { button: 0 })
  fireEvent.mouseUp(element, { button: 0 })
  fireEvent.click(element, { button: 0 })
}

function renderDialogWithToast(onOpenChange: (open: boolean) => void) {
  return render(
    <>
      <NotificationStack />
      <Dialog onOpenChange={onOpenChange} open>
        <DialogContent>
          <DialogTitle>New task</DialogTitle>
          <textarea aria-label="draft" defaultValue="unsaved work" />
        </DialogContent>
      </Dialog>
    </>
  )
}

describe('toasts over an open dialog (#135698)', () => {
  beforeEach(() => clearNotifications())

  afterEach(() => {
    cleanup()
    clearNotifications()
  })

  it.each(['default', 'bottom-right'] as const)(
    'dismissing a %s toast does not close the dialog underneath',
    async placement => {
      const onOpenChange = vi.fn()
      renderDialogWithToast(onOpenChange)
      act(() => {
        notify({ id: 'blocked', title: 'Task blocked', message: 'needs your input', placement, durationMs: 0 })
      })
      await flushTimers()

      // `hidden: true`: the modal dialog aria-hides its body-level siblings.
      press(screen.getByRole('button', { hidden: true, name: /Dismiss/ }))
      await flushTimers()

      // The toast leaves through an exit animation.
      await waitFor(() => expect(screen.queryByText('needs your input')).toBeNull())
      expect(onOpenChange).not.toHaveBeenCalledWith(false)
      expect((screen.getByLabelText('draft') as HTMLTextAreaElement).value).toBe('unsaved work')
    }
  )

  it('clicking the body of a toast does not close the dialog either', async () => {
    const onOpenChange = vi.fn()
    renderDialogWithToast(onOpenChange)
    act(() => {
      notify({ id: 'blocked', title: 'Task blocked', message: 'needs your input', durationMs: 0 })
    })
    await flushTimers()

    press(screen.getByText('needs your input'))
    await flushTimers()

    expect(onOpenChange).not.toHaveBeenCalledWith(false)
  })

  it('still closes the dialog on a genuine overlay click', async () => {
    const onOpenChange = vi.fn()
    renderDialogWithToast(onOpenChange)
    act(() => {
      notify({ id: 'blocked', title: 'Task blocked', message: 'needs your input', durationMs: 0 })
    })
    await flushTimers()

    const overlay = document.querySelector('[data-slot="dialog-overlay"]') as HTMLElement
    fireEvent.pointerDown(overlay, { button: 0 })
    fireEvent.pointerUp(overlay, { button: 0 })
    fireEvent.click(overlay, { button: 0 })
    await flushTimers()

    expect(onOpenChange).toHaveBeenCalledWith(false)
  })

  it('keeps a caller-supplied onInteractOutside working', async () => {
    const onOpenChange = vi.fn()
    const onInteractOutside = vi.fn((event: Event) => event.preventDefault())
    render(
      <Dialog onOpenChange={onOpenChange} open>
        <DialogContent onInteractOutside={onInteractOutside}>
          <DialogTitle>Guarded</DialogTitle>
        </DialogContent>
      </Dialog>
    )
    await flushTimers()

    const overlay = document.querySelector('[data-slot="dialog-overlay"]') as HTMLElement
    fireEvent.pointerDown(overlay, { button: 0 })
    fireEvent.pointerUp(overlay, { button: 0 })
    fireEvent.click(overlay, { button: 0 })
    await flushTimers()

    expect(onInteractOutside).toHaveBeenCalled()
    expect(onOpenChange).not.toHaveBeenCalledWith(false)
  })
})

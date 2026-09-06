import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ConfirmDialog } from './confirm-dialog'

afterEach(cleanup)

describe('ConfirmDialog secondary action', () => {
  function renderWithSecondary() {
    const onConfirm = vi.fn()
    const onClose = vi.fn()
    const onSecondary = vi.fn()

    render(
      <ConfirmDialog
        onClose={onClose}
        onConfirm={onConfirm}
        open
        secondaryAction={{ label: 'Remove from sidebar', onClick: onSecondary }}
        title="Remove worktree?"
      />
    )

    return { onClose, onConfirm, onSecondary }
  }

  it('runs the secondary action and closes without confirming', async () => {
    const { onClose, onConfirm, onSecondary } = renderWithSecondary()

    fireEvent.click(await screen.findByRole('button', { name: 'Remove from sidebar' }))

    expect(onSecondary).toHaveBeenCalledTimes(1)
    expect(onClose).toHaveBeenCalledTimes(1)
    expect(onConfirm).not.toHaveBeenCalled()
  })

  it('still opens focused on Confirm, so Enter confirms rather than picking the secondary', async () => {
    const { onConfirm, onSecondary } = renderWithSecondary()

    const dialog = await screen.findByRole('dialog')

    // eslint-disable-next-line no-restricted-globals -- asserting real focus requires the live document
    await waitFor(() => expect(dialog.contains(document.activeElement)).toBe(true))
    // eslint-disable-next-line no-restricted-globals -- asserting real focus requires the live document
    fireEvent.keyDown(document.activeElement!, { key: 'Enter' })

    await waitFor(() => expect(onConfirm).toHaveBeenCalledTimes(1))
    expect(onSecondary).not.toHaveBeenCalled()
  })
})

it.each(['wrong', 'correct', 'cancel', 'reopen', 'space', 'duplicate'])(
  'requires the exact typed intent before confirming: %s',
  async mode => {
    const onConfirm = vi.fn()
    const onClose = vi.fn()
    const props = { open: true, title: 'Restart', typedConfirmation: 'RESTART', onConfirm, onClose: onClose }
    const view = render(<ConfirmDialog {...props} />)
    const input = await screen.findByRole('textbox')
    const confirm = await screen.findByRole('button', { name: 'Confirm' })
    await waitFor(() => expect(input.matches(':focus')).toBe(true))
    fireEvent.click(confirm)
    fireEvent.keyDown(input, { key: 'Enter' })
    expect(onConfirm).not.toHaveBeenCalled()

    if (mode === 'cancel') {
      fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
      expect(onClose).toHaveBeenCalledTimes(1)

      return
    }

    fireEvent.change(input, { target: { value: mode === 'wrong' ? 'restart' : 'RESTART' } })

    if (mode === 'space') {
      const event = new KeyboardEvent('keydown', { key: ' ', bubbles: true, cancelable: true })
      input.dispatchEvent(event)
      expect(event.defaultPrevented).toBe(false)
      expect(onConfirm).not.toHaveBeenCalled()
    } else if (mode === 'duplicate') {
      fireEvent.click(confirm)
      fireEvent.click(confirm)
      fireEvent.keyDown(input, { key: 'Enter' })
      await waitFor(() => expect(onConfirm).toHaveBeenCalledTimes(1))
    } else if (mode === 'reopen') {
      view.rerender(<ConfirmDialog {...props} open={false} />)
      view.rerender(<ConfirmDialog {...props} />)
      await waitFor(() => expect((screen.getByRole('textbox') as HTMLInputElement).value).toBe(''))
      fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter' })
      expect(onConfirm).not.toHaveBeenCalled()
    } else {
      fireEvent.keyDown(input, { key: 'Enter' })
      await waitFor(() => expect(onConfirm).toHaveBeenCalledTimes(mode === 'correct' ? 1 : 0))
    }
  }
)

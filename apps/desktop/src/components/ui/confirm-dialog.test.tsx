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

  it('keeps keys on the focused secondary action', async () => {
    const { onConfirm, onSecondary } = renderWithSecondary()
    const secondary = await screen.findByRole('button', { name: 'Remove from sidebar' })
    secondary.focus()
    fireEvent.keyDown(secondary, { key: 'Enter' })
    fireEvent.keyDown(secondary, { key: ' ' })
    expect(onConfirm).not.toHaveBeenCalled()
    fireEvent.click(secondary)
    expect(onSecondary).toHaveBeenCalledTimes(1)
  })
})

describe('ConfirmDialog keyboard and focus', () => {
  it('focuses Cancel for a destructive action and leaves its keys to the button', async () => {
    const onConfirm = vi.fn()
    const onClose = vi.fn()
    render(<ConfirmDialog destructive onClose={onClose} onConfirm={onConfirm} open title="Delete worktree?" />)

    const cancel = await screen.findByRole('button', { name: 'Cancel' })
    // eslint-disable-next-line no-restricted-globals -- asserting real focus requires the live document
    await waitFor(() => expect(document.activeElement).toBe(cancel))
    fireEvent.keyDown(cancel, { key: 'Enter' })
    fireEvent.keyDown(cancel, { key: ' ' })
    expect(onConfirm).not.toHaveBeenCalled()
    fireEvent.click(cancel)
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('does not confirm from an input or from a focused Confirm keydown before native click', async () => {
    const onConfirm = vi.fn()
    render(
      <ConfirmDialog onClose={vi.fn()} onConfirm={onConfirm} open title="Confirm?">
        <input aria-label="Reason" />
      </ConfirmDialog>
    )

    const input = await screen.findByRole('textbox', { name: 'Reason' })
    input.focus()
    fireEvent.keyDown(input, { key: 'Enter' })
    fireEvent.keyDown(input, { key: ' ' })
    expect(onConfirm).not.toHaveBeenCalled()

    const confirm = screen.getByRole('button', { name: 'Confirm' })
    confirm.focus()
    fireEvent.keyDown(confirm, { key: 'Enter' })
    expect(onConfirm).not.toHaveBeenCalled()
    fireEvent.click(confirm)
    await waitFor(() => expect(onConfirm).toHaveBeenCalledTimes(1))
  })

  it('announces an async failure once and allows a retry', async () => {
    const onConfirm = vi.fn().mockRejectedValueOnce(new Error('Could not delete')).mockResolvedValueOnce(undefined)
    render(<ConfirmDialog onClose={vi.fn()} onConfirm={onConfirm} open title="Delete worktree?" />)
    fireEvent.click(await screen.findByRole('button', { name: 'Confirm' }))
    expect((await screen.findByRole('alert')).textContent).toContain('Could not delete')
    expect(screen.getAllByRole('alert')).toHaveLength(1)
    fireEvent.click(screen.getByRole('button', { name: 'Confirm' }))
    await waitFor(() => expect(onConfirm).toHaveBeenCalledTimes(2))
  })
})

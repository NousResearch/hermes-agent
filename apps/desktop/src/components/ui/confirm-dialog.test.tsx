import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { act } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $shellChoice } from '@/store/shell-variant'

import { ConfirmDialog } from './confirm-dialog'

afterEach(() => {
  cleanup()
  $shellChoice.set(null)
})

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

describe('ConfirmDialog destructive confirm', () => {
  function renderDestructive() {
    const onConfirm = vi.fn()

    render(
      <ConfirmDialog
        confirmLabel="Delete"
        destructive
        onClose={vi.fn()}
        onConfirm={onConfirm}
        open
        title="Delete session?"
      />
    )

    return { onConfirm }
  }

  it('opens on Cancel in the redesign, so a stray Enter never destroys', async () => {
    const { onConfirm } = renderDestructive()
    const cancel = await screen.findByRole('button', { name: 'Cancel' })

    // eslint-disable-next-line no-restricted-globals -- asserting real focus requires the live document
    await waitFor(() => expect(document.activeElement).toBe(cancel))
    fireEvent.keyDown(cancel, { key: 'Enter' })

    expect(onConfirm).not.toHaveBeenCalled()
    expect(screen.queryByRole('button', { name: 'Close' })).toBeNull()
  })

  it('keeps classic opening on the destructive action, with its corner close', async () => {
    act(() => $shellChoice.set(false))
    const { onConfirm } = renderDestructive()
    const confirm = await screen.findByRole('button', { name: 'Delete' })

    // eslint-disable-next-line no-restricted-globals -- asserting real focus requires the live document
    await waitFor(() => expect(document.activeElement).toBe(confirm))
    fireEvent.keyDown(confirm, { key: 'Enter' })

    await waitFor(() => expect(onConfirm).toHaveBeenCalledTimes(1))
    expect(screen.getAllByRole('button', { name: 'Close' }).length).toBeGreaterThan(0)
  })
})

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

describe('ConfirmDialog async announcements', () => {
  it('announces a rejected confirmation as an alert', async () => {
    render(
      <ConfirmDialog
        onClose={vi.fn()}
        onConfirm={() => Promise.reject(new Error('Task changed'))}
        open
        title="Update task?"
      />
    )

    fireEvent.click(await screen.findByRole('button', { name: 'Confirm' }))

    expect((await screen.findByRole('alert')).textContent).toContain('Task changed')
  })

  it('announces pending and completed labels through the confirm action', async () => {
    render(
      <ConfirmDialog
        busyLabel="Updating task"
        doneLabel="Task updated"
        onClose={vi.fn()}
        onConfirm={() => Promise.resolve()}
        open
        title="Update task?"
      />
    )

    const confirm = await screen.findByRole('button', { name: 'Confirm' })
    expect(confirm.getAttribute('aria-live')).toBe('polite')
    fireEvent.click(confirm)

    await waitFor(() => expect(confirm.textContent).toContain('Task updated'))
  })
})

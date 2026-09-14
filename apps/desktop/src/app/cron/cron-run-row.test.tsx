import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { makeSessionInfo } from '@/test/session-info'

import { CronRunRow } from './cron-run-row'

afterEach(cleanup)

function openActions() {
  const trigger = screen.getByRole('button', { name: 'Session actions' })

  fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
  fireEvent.pointerUp(trigger, { button: 0, pointerType: 'mouse' })
  fireEvent.click(trigger)
}

function renderRow(variant: 'detail' | 'sidebar') {
  const onDelete = vi.fn().mockResolvedValue(true)
  const onDeleted = vi.fn()
  const onOpen = vi.fn()
  const run = makeSessionInfo({ id: 'cron-run-1', profile: 'work', title: 'Nightly report' })

  render(
    <I18nProvider configClient={null}>
      <CronRunRow
        active={variant === 'sidebar'}
        onDelete={onDelete}
        onDeleted={onDeleted}
        onOpen={onOpen}
        run={run}
        time="Aug 21, 10:00"
        variant={variant}
      />
    </I18nProvider>
  )

  return { onDelete, onDeleted, onOpen }
}

describe('CronRunRow', () => {
  it('names the sidebar run button with both its title and time', () => {
    renderRow('sidebar')

    expect(screen.getByRole('button', { name: 'Nightly report — Aug 21, 10:00' })).toBeTruthy()
  })

  it.each(['detail', 'sidebar'] as const)(
    'deletes one %s run after confirmation and preserves its profile',
    async variant => {
      const { onDelete, onDeleted } = renderRow(variant)

      openActions()
      fireEvent.click(await screen.findByRole('menuitem', { name: 'Delete' }))

      const dialog = await screen.findByRole('dialog')
      expect(within(dialog).getByText(/Nightly report/)).toBeTruthy()
      expect(onDelete).not.toHaveBeenCalled()

      fireEvent.click(screen.getByRole('button', { name: 'Delete' }))

      expect(await screen.findByText('Session deleted')).toBeTruthy()
      expect(onDelete).toHaveBeenCalledWith('cron-run-1', 'work')
      expect(onDeleted).toHaveBeenCalledWith('cron-run-1')
    }
  )

  it('keeps the row when session deletion reports failure', async () => {
    const { onDelete, onDeleted } = renderRow('detail')
    onDelete.mockResolvedValue(false)

    openActions()
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Delete' }))
    fireEvent.click(await screen.findByRole('button', { name: 'Delete' }))

    expect(await screen.findByText('Delete failed')).toBeTruthy()
    expect(onDeleted).not.toHaveBeenCalled()
    expect(screen.getByText('Nightly report')).toBeTruthy()
  })
})

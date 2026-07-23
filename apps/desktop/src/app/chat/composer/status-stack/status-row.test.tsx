import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { StatusItemRow } from './status-row'

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      statusStack: {
        running: 'Running',
        stop: 'Stop',
        dismiss: 'Dismiss',
        exit: (code: number) => `exit ${code}`
      }
    }
  })
}))

describe('StatusItemRow', () => {
  afterEach(() => {
    cleanup()
  })

  it('shows the full todo title in a hover tooltip', async () => {
    const title = 'A long todo title that would be truncated by max-w-[18rem] truncate'

    render(
      <StatusItemRow
        item={{
          id: 'todo-1',
          title,
          state: 'running',
          type: 'todo',
          todoStatus: 'in_progress'
        }}
      />
    )

    fireEvent.pointerMove(screen.getByText(title), { pointerType: 'mouse' })
    const tooltip = await screen.findByRole('tooltip')

    expect(tooltip.textContent).toContain(title)
  })
})

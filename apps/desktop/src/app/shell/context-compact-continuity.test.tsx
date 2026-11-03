import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { ContextUsagePanel } from './context-usage-panel'

// Applies to ordinary chats and to Bot Mode's permanent Bot Chat alike.
afterEach(cleanup)

it('offers compaction with an explanation that the conversation continues here', () => {
  const onRequest = vi.fn()

  render(
    <ContextUsagePanel
      breakdown={null}
      compact={{
        busy: false,
        confirming: false,
        error: null,
        onCancel: vi.fn(),
        onConfirm: vi.fn(),
        onRequest,
        pending: false,
        unavailable: false
      }}
      loading={false}
      usage={{ calls: 0, context_max: 160_000, context_used: 125_000, input: 0, output: 0, total: 0 }}
    />
  )

  expect(screen.getByText('Summarize older context to make room. Continue in this same conversation.')).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: 'Compact conversation' }))
  expect(onRequest).toHaveBeenCalledTimes(1)
})

it('hides confirmation and explains why when a reply starts', () => {
  const onConfirm = vi.fn()

  render(
    <ContextUsagePanel
      breakdown={null}
      compact={{
        busy: true,
        confirming: true,
        error: null,
        onCancel: vi.fn(),
        onConfirm,
        onRequest: vi.fn(),
        pending: false,
        unavailable: false
      }}
      loading={false}
      usage={{ calls: 0, input: 0, output: 0, total: 0 }}
    />
  )

  const confirm = screen.getByRole('button', { name: 'Compact conversation' }) as HTMLButtonElement

  expect(screen.queryByRole('dialog')).toBeNull()
  expect(confirm.disabled).toBe(true)
  expect(screen.getByText('Finish or stop the current reply first')).toBeTruthy()
  fireEvent.click(confirm)
  expect(onConfirm).not.toHaveBeenCalled()
})

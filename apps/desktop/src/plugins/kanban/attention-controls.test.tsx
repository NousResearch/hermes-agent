import { host } from '@hermes/plugin-sdk'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { ReactElement } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as KanbanApi from './api'
import { updateAttention } from './api'
import { AttentionControls } from './board'
import { formatLocalDateTime, parseLocalDateTime } from './datetime-local'
import type { AttentionReceipt, KanbanTask } from './types'

vi.mock('./api', async importOriginal => {
  const actual = await importOriginal<typeof KanbanApi>()

  return { ...actual, updateAttention: vi.fn() }
})

const updateAttentionMock = vi.mocked(updateAttention)

const baseTask: KanbanTask = { id: 'task-safe', status: 'running', title: 'Privacy-safe evidence task' }

const receipt = (state: AttentionReceipt['state'], revision: number): AttentionReceipt => ({
  reason: 'receipt',
  revision,
  state,
  wake_at: state === 'snoozed' ? 1_800_000_000 : null
})

const response = (state: AttentionReceipt['state'], revision: number) => ({ attention: receipt(state, revision), idempotent: false })

function view(task: KanbanTask): ReactElement {
  return (
    <QueryClientProvider client={new QueryClient({ defaultOptions: { mutations: { retry: false }, queries: { retry: false } } })}>
      <AttentionControls task={task} />
    </QueryClientProvider>
  )
}

afterEach(cleanup)

beforeEach(() => {
  updateAttentionMock.mockReset()
  vi.spyOn(host, 'notify').mockImplementation(() => '')
})

describe('attention control lifecycle accessibility', () => {
  it('keeps one live region through settle, settled, failed wake, and successful wake', async () => {
    updateAttentionMock.mockResolvedValueOnce(response('settled', 1))
    const rendered = render(view({ ...baseTask, attention: receipt('active', 0) }))

    expect(screen.getAllByRole('status')).toHaveLength(1)
    fireEvent.click(screen.getByRole('button', { name: 'Settle' }))
    await waitFor(() => expect(screen.getByRole('status').textContent).toBe('Attention settled'))

    rendered.rerender(view({ ...baseTask, attention: receipt('settled', 1) }))
    expect(screen.getAllByRole('status')).toHaveLength(1)
    expect((screen.getByRole('button', { name: 'Wake' }) as HTMLButtonElement).disabled).toBe(false)

    updateAttentionMock.mockRejectedValueOnce(new Error('Wake was not accepted'))
    fireEvent.click(screen.getByRole('button', { name: 'Wake' }))
    await waitFor(() => expect(screen.getByRole('status').textContent).toBe('Wake was not accepted'))

    updateAttentionMock.mockResolvedValueOnce(response('active', 2))
    fireEvent.click(screen.getByRole('button', { name: 'Wake' }))
    await waitFor(() => expect(screen.getByRole('status').textContent).toBe('Task awake'))
    expect(updateAttentionMock).toHaveBeenLastCalledWith('task-safe', 'wake', 1, undefined)
  })

  it('supports keyboard activation and touch-sized snooze controls without duplicating status', async () => {
    updateAttentionMock.mockResolvedValueOnce(response('snoozed', 1))
    render(view({ ...baseTask, attention: receipt('active', 0) }))

    const disclosure = screen.getByText('Snooze…')
    fireEvent.keyDown(disclosure, { key: 'Enter' })
    fireEvent.click(screen.getByRole('button', { name: '1 hour' }))

    await waitFor(() => expect(screen.getByRole('status').textContent).toBe('Task snoozed'))
    expect(screen.getAllByRole('status')).toHaveLength(1)
    expect(screen.getByRole('button', { name: '1 hour' }).classList.contains('min-h-8')).toBe(true)
  })
})

describe('attention local wake fields', () => {
  it.each([
    ['UTC', '2026-01-02T03:04'],
    ['America/Chicago', '2026-01-01T21:04'],
    ['Asia/Kathmandu', '2026-01-02T08:49']
  ])('formats local wall time in %s', (tz, expected) => {
    process.env.TZ = tz
    expect(formatLocalDateTime(new Date('2026-01-02T03:04:00Z'))).toBe(expected)
  })

  it('rejects normalized calendar values and DST gaps', () => {
    process.env.TZ = 'America/Chicago'
    expect(parseLocalDateTime('2026-02-30T09:00')).toBeNull()
    expect(parseLocalDateTime('2026-03-08T02:30')).toBeNull()
  })

  it('uses the earlier DST-fold occurrence and preserves its wall-clock field', () => {
    process.env.TZ = 'America/Chicago'
    const folded = parseLocalDateTime('2026-11-01T01:30')
    expect(folded).not.toBeNull()
    expect(formatLocalDateTime(folded!)).toBe('2026-11-01T01:30')
    expect(folded!.toISOString()).toBe('2026-11-01T06:30:00.000Z')
  })
})

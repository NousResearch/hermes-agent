// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { WisdomSyncStatus } from './wisdom-sync-status'
const { read, retry } = vi.hoisted(() => ({ read: vi.fn(), retry: vi.fn() }))
vi.mock('@/hermes', () => ({ getWisdomSync: read, retryWisdomSync: retry }))
vi.mock('@/api/client', () => ({ capabilityScoped: (p: string) => p, profileScopeKey: (p: string) => p }))
vi.mock('@/i18n', () => ({
  useI18n: () => ({ t: { common: { refresh: 'Refresh', loading: 'Loading' }, skills: { collective: {} } } })
}))
const counts = { pending: 0, syncing: 0, retryable: 0, conflict: 0, uncertain: 0, waiting_for_receipt: 0 }
const snapshot = { delivery: counts, operation: { ...counts, retryable: 1 }, can_retry: true }
beforeEach(() => {
  vi.clearAllMocks()
  read.mockResolvedValue(snapshot)
  retry.mockResolvedValue({ delivery: counts, operation: counts, can_retry: false })
})
afterEach(cleanup)
it('opens read-only and only retries on the explicit control', async () => {
  render(<WisdomSyncStatus profile="research" />)
  expect(read).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('button', { name: 'Receipt and report sync' }))
  await screen.findByText('Retry available: 1')
  expect(read).toHaveBeenCalledWith('research')
  expect(retry).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('button', { name: 'Retry sync' }))
  await waitFor(() => expect(screen.getAllByText('Up to date')).toHaveLength(2))
  expect(retry).toHaveBeenCalledExactlyOnceWith('research')
  expect(screen.getByRole('button', { name: 'Retry sync' }).hasAttribute('disabled')).toBe(true)
})
it('does not expose a retired profiles in-flight result', async () => {
  let finish!: (v: typeof snapshot) => void
  read.mockReturnValueOnce(
    new Promise(resolve => {
      finish = resolve
    })
  )
  const view = render(<WisdomSyncStatus profile="research" />)
  fireEvent.click(screen.getByRole('button', { name: 'Receipt and report sync' }))
  view.rerender(<WisdomSyncStatus profile="other" />)
  await act(async () => {
    finish(snapshot)
  })
  expect(screen.queryByText('Retry available: 1')).toBeNull()
  expect(screen.getByRole('button', { name: 'Receipt and report sync' }).getAttribute('aria-expanded')).toBe('false')
})
it('shows unavailable without raw errors or an enabled retry', async () => {
  read.mockRejectedValueOnce(new Error('secret transport payload'))
  render(<WisdomSyncStatus />)
  fireEvent.click(screen.getByRole('button', { name: 'Receipt and report sync' }))
  expect((await screen.findByRole('alert')).textContent).toContain('Sync status could not be confirmed.')
  expect(screen.queryByText('secret transport payload')).toBeNull()
  expect(screen.getByRole('button', { name: 'Retry sync' }).hasAttribute('disabled')).toBe(true)
})

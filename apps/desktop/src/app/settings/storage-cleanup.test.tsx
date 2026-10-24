import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

const { confirmMock, rpc } = vi.hoisted(() => ({
  confirmMock: vi.fn(),
  rpc: vi.fn()
}))

vi.mock('@/store/confirm', () => ({ confirm: confirmMock }))
vi.mock('@/store/gateway', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  requestGatewayForAgent: rpc
}))

import { StorageCleanup } from './storage-cleanup'

afterEach((): void => {
  cleanup()
  vi.unstubAllGlobals()
  confirmMock.mockReset()
  rpc.mockReset()
})

it('deletes only what the dry run previewed, through the owner, after confirmation', async (): Promise<void> => {
  const api = vi.fn().mockResolvedValue({
    matched: 3,
    skipped_open: 1,
    oldest_last_active: 1_700_000_000,
    newest_last_active: 1_710_000_000
  })

  vi.stubGlobal('hermesDesktop', { api })
  rpc.mockResolvedValue({ deleted: 3 })
  const onFreed = vi.fn()

  render(
    <I18nProvider>
      <StorageCleanup onFreed={onFreed} />
    </I18nProvider>
  )

  await waitFor((): void => expect(screen.getByText(/3 ended sessions unused for 90 days/)).toBeTruthy())
  // The preview is a read-only dry run with the CLI's default window.
  expect(api.mock.calls[0][0]).toMatchObject({
    path: '/api/sessions/prune',
    method: 'POST',
    body: { older_than_days: 90, dry_run: true }
  })

  // Declining the confirmation deletes nothing.
  confirmMock.mockResolvedValueOnce(false)
  fireEvent.click(screen.getByRole('button', { name: /Delete/ }))
  await waitFor((): void => expect(confirmMock).toHaveBeenCalledTimes(1))
  expect(rpc).not.toHaveBeenCalled()

  confirmMock.mockResolvedValueOnce(true)
  fireEvent.click(screen.getByRole('button', { name: /Delete/ }))
  await waitFor((): void => expect(onFreed).toHaveBeenCalledTimes(1))
  expect(confirmMock.mock.calls[1][0]).toMatchObject({ destructive: true, title: expect.stringContaining('3') })
  // The live owner deletes (a second writer under the gateway is refused), with the same filters.
  expect(rpc.mock.calls[0].slice(2, 4)).toEqual([
    'session.prune',
    { filters: { older_than_days: 90, archived: false } }
  ])
})

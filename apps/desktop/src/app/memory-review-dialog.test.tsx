import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import { I18nProvider } from '@/i18n'
import { zh } from '@/i18n/zh'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection } from '@/store/session'
import { MemoryReviewDialog } from './memory-review-dialog'
import { $memoryReview, openMemoryReview } from '@/store/memory-review'

afterEach(() => {
  cleanup()
  $memoryReview.set(null)
})

it('renders translated review controls in the active locale', async () => {
  act(() => openMemoryReview(vi.fn().mockResolvedValue({ write_approval: true, batches: [] }), 'generic-runtime'))
  render(
    <I18nProvider initialLocale="zh" configClient={null}>
      <MemoryReviewDialog />
    </I18nProvider>
  )
  await screen.findByText(zh.memoryReview.empty)
  expect(screen.getByRole('heading', { name: zh.memoryReview.title })).toBeTruthy()
  expect(screen.getByRole('button', { name: zh.memoryReview.refresh })).toBeTruthy()
  expect(screen.getByText(zh.memoryReview.gateOn)).toBeTruthy()
})

it('closes on profile switch and ignores an old loading result', async () => {
  let finish!: (value: unknown) => void
  const request = vi.fn(
    () =>
      new Promise(resolve => {
        finish = resolve
      })
  )
  act(() => openMemoryReview(request as never, 'old-runtime'))
  render(<MemoryReviewDialog />)
  await screen.findByRole('status')
  act(() => $activeGatewayProfile.set('generic-other-profile'))
  expect(screen.queryByRole('dialog')).toBeNull()
  await act(async () => finish({ write_approval: false, batches: [] }))
  expect(screen.queryByRole('dialog')).toBeNull()
})

it('connection switches close the review', () => {
  act(() => openMemoryReview(vi.fn() as never, 'generic-session'))
  act(() =>
    $connection.set({ ...$connection.get(), connectionId: 'generic-other-connection' } as NonNullable<
      ReturnType<typeof $connection.get>
    >)
  )
  expect($memoryReview.get()).toBeNull()
})

it('shows full old/new diff and decides an atomic batch on the captured session', async () => {
  const batch = {
    id: 'abcdef12',
    summary: 'Generic batch',
    origin: 'background_review',
    created_at: 1,
    target: 'memory',
    action: 'batch',
    operation_count: 2,
    before: 'Generic old clause.',
    after: 'Generic new clause.',
    diff: '--- a/MEMORY.md\n+++ b/MEMORY.md\n@@ -1 +1 @@\n-Generic old clause.\n+Generic new clause.',
    revision: 'generic-revision',
    can_approve: true,
    error: ''
  }
  const request = vi
    .fn()
    .mockResolvedValueOnce({ write_approval: false, batches: [batch] })
    .mockResolvedValueOnce({ success: true, error: '' })
    .mockResolvedValue({ write_approval: false, batches: [] })
  act(() => openMemoryReview(request, 'focused-runtime'))
  render(<MemoryReviewDialog />)
  await screen.findByText('Generic old clause.')
  act(() => screen.getByRole('button', { name: 'Raw unified diff' }).click())
  expect(screen.getByTestId('memory-raw-diff').textContent).toBe(batch.diff)
  act(() => screen.getByRole('button', { name: 'Formatted diff' }).click())
  expect(screen.getByText('Generic new clause.')).toBeTruthy()
  expect(screen.getByText(/off/i)).toBeTruthy()
  act(() => screen.getByRole('button', { name: 'Approve' }).click())
  await waitFor(() =>
    expect(request).toHaveBeenCalledWith('memory.decide', {
      session_id: 'focused-runtime',
      id: batch.id,
      decision: 'approve',
      revision: batch.revision
    })
  )
  await screen.findByText('No pending memory writes.')
})

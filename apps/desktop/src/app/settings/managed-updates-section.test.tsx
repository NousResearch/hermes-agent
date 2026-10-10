import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, expect, it, vi } from 'vitest'

import type { DesktopManagedConnectionUpdateResult } from '@/global'
import { ko } from '@/i18n/ko'
import { $managedUpdates, _resetManagedUpdatesForTests } from '@/store/managed-updates'

import { ManagedUpdatesSection } from './managed-updates-section'

vi.mock('@/i18n', async () => {
  const { ko } = await import('@/i18n/ko')

  return { useI18n: () => ({ t: ko }) }
})
vi.mock('@/store/connections', async () => ({
  $connectionsRegistry: (await import('nanostores')).atom({
    connections: [{ id: 'test-ssh', kind: 'ssh', label: '검증 연결', host: 'test.invalid' }]
  })
}))

const updateManaged = vi.fn<() => Promise<DesktopManagedConnectionUpdateResult>>()

beforeEach(() => {
  _resetManagedUpdatesForTests()
  updateManaged.mockReset()
  ;(window as { hermesDesktop?: unknown }).hermesDesktop = { connections: { updateManaged } }
})

it('keeps both update and restore failure reasons inspectable alongside truthful Korean statuses', async () => {
  const updateReason = 'checkout rejected: a local edit would be overwritten'
  const restoreReason = 'SSH forwarding could not be re-established'
  updateManaged.mockResolvedValue({
    connectionId: 'test-ssh',
    correlationId: 'test-run',
    exitCode: 1,
    ok: false,
    updateOk: false,
    restoreOk: false,
    outcome: 'update-and-restore-failed',
    message: 'The update failed and some profiles could not be restored.',
    error: updateReason,
    receipt: null,
    scopes: [{ profile: '작업', restored: false, error: restoreReason }]
  })

  render(<ManagedUpdatesSection />)
  fireEvent.click(screen.getByRole('button', { name: ko.settings.managedUpdates.update }))

  await waitFor(() => expect(screen.getByText(updateReason)).toBeTruthy())
  expect(screen.getByText(ko.settings.managedUpdates.failed)).toBeTruthy()
  expect(screen.getByText(ko.settings.managedUpdates.scopeNotRestored('작업', restoreReason))).toBeTruthy()
  const details = screen.getByText(updateReason).closest('details')
  expect(details).not.toBeNull()
  expect(details?.querySelector('summary')?.textContent).toBe(ko.notifications.details)
})

it('localizes known receipt outcomes only at display time and preserves unfamiliar outcomes', async () => {
  render(<ManagedUpdatesSection />)

  const outcomes = [
    ...Object.entries(ko.settings.managedUpdates.receiptOutcomes),
    ['future-updater-result', 'future-updater-result'],
    ['constructor', 'constructor']
  ]

  for (const [outcome, label] of outcomes) {
    const receipt = { correlationId: 'receipt-1234', outcome }
    updateManaged.mockResolvedValue({
      connectionId: 'test-ssh',
      correlationId: receipt.correlationId,
      exitCode: 0,
      ok: true,
      updateOk: true,
      restoreOk: true,
      outcome: 'updated',
      receipt,
      scopes: []
    })
    fireEvent.click(screen.getByRole('button', { name: ko.settings.managedUpdates.update }))

    await waitFor(() =>
      expect(
        screen.getByText(ko.settings.managedUpdates.receipt(receipt.correlationId.slice(0, 8), label))
      ).toBeTruthy()
    )
    expect($managedUpdates.get()['test-ssh'].receipt?.outcome).toBe(outcome)
    expect(receipt.outcome).toBe(outcome)
  }
})

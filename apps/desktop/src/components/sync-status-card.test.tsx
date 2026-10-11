import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import type { DesktopSyncReceipt } from '@/global'

import { SyncStatusCard } from './sync-status-card'

afterEach((): void => {
  cleanup()
  vi.unstubAllGlobals()
})

const details: DesktopSyncReceipt = {
  plugin_bisect: [
    { plugin: 'conflictor', action: 'disabled', reason: 'conflicts with core pin' },
    { plugin: 'kept', action: 'kept', reason: '' }
  ],
  plugin_checks: [
    { name: 'bad-url', needs_fixing: 'update_url points at a fork', update_available: false },
    { name: 'grower', update_available: true, current: '1.0.0', latest: '1.1.0' },
    { name: 'fine', update_available: false }
  ]
}

interface ReceiptCase {
  name: string
  receipt: DesktopSyncReceipt | null
  headline: RegExp | null
  tone?: string
}

const cases: ReceiptCase[] = [
  { name: 'missing', receipt: null, headline: null },
  { name: 'empty', receipt: {}, headline: null },
  {
    name: 'healthy no-op',
    receipt: { outcome: 'ok', venv_rebuild: { ok: false, reason: 'already in sync' } },
    headline: null
  },
  {
    name: 'unknown availability',
    receipt: { plugin_checks: [{ name: 'mystery', update_available: null }] },
    headline: null
  },
  {
    name: 'embedded failed',
    receipt: {
      ...details,
      outcome: 'ok',
      pm_sync_outcome: 'failed',
      pm_steps: [{ name: 'sync', ok: false, detail: 'network unavailable' }]
    },
    headline: /rebuild failed — network unavailable/,
    tone: 'text-destructive'
  },
  {
    name: 'embedded refused',
    receipt: { ...details, pm_sync_outcome: 'refused' },
    headline: /rebuild failed/,
    tone: 'text-destructive'
  },
  {
    name: 'legacy failed',
    receipt: { ...details, outcome: 'failed', venv_rebuild: { ok: false, reason: 'uv sync exited 1' } },
    headline: /rebuild failed — uv sync exited 1/,
    tone: 'text-destructive'
  },
  {
    name: 'review beats bisect and updates',
    receipt: details,
    headline: /need update-url review/,
    tone: 'text-amber-600'
  },
  {
    name: 'bisect beats updates',
    receipt: { ...details, plugin_checks: details.plugin_checks?.slice(1) },
    headline: /disabled by dependency conflicts/,
    tone: 'text-amber-600'
  },
  {
    name: 'updates only',
    receipt: { plugin_checks: details.plugin_checks?.slice(1) },
    headline: /Plugin updates available/,
    tone: 'text-muted-foreground'
  }
]

it.each(cases)(
  '$name receipt paints the dominant warning and subordinate rows',
  async ({ receipt, headline, tone }: ReceiptCase): Promise<void> => {
    vi.stubGlobal('hermesDesktop', {
      getSyncStatus: async (): Promise<DesktopSyncReceipt | null> => receipt
    } satisfies Pick<Window['hermesDesktop'], 'getSyncStatus'>)
    await act(async (): Promise<void> => {
      render(<SyncStatusCard />)
    })

    if (!headline) {
      expect(screen.queryByTestId('sync-status-card')).toBeNull()

      return
    }

    expect(screen.getByText(headline).className).toContain(tone)

    if (receipt?.plugin_bisect) {
      expect(screen.getByText('conflictor: conflicts with core pin')).toBeTruthy()
    }

    if (receipt?.plugin_checks?.some(row => row.needs_fixing)) {
      expect(screen.getByText('bad-url: update_url points at a fork')).toBeTruthy()
    }

    expect(screen.getByText('grower: 1.0.0 → 1.1.0')).toBeTruthy()
    expect(screen.queryByText('fine')).toBeNull()
    expect(screen.queryByText('kept')).toBeNull()
  }
)

it('a rejected bridge leaves the overlay usable', async (): Promise<void> => {
  vi.stubGlobal('hermesDesktop', {
    getSyncStatus: async (): Promise<never> => {
      throw new Error('bridge gone')
    }
  } satisfies Pick<Window['hermesDesktop'], 'getSyncStatus'>)
  await act(async (): Promise<void> => {
    render(<SyncStatusCard />)
  })
  expect(screen.queryByTestId('sync-status-card')).toBeNull()
})

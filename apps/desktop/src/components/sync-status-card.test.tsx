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
  tail?: { message: string; at?: string | null } | null
  headline: RegExp | null
  tone?: string
}

const updateBase: DesktopSyncReceipt = {
  outcome: 'failed',
  started_at: '2026-10-20T10:00:00.000Z',
  finished_at: '2026-10-20T10:09:00.000Z',
  pre_update: { sha: 'abc123def456', short_sha: 'abc123', version: '0.21.5', source: 'git' },
  steps: [{ name: 'sibling_profile_snapshots', ok: true, detail: 'ain=ok' }]
}

const freshTail = { message: 'Building update products', at: new Date(Date.now() - 30_000).toISOString() }
const staleTail = { message: 'Building update products', at: '2026-10-01T10:00:00.000Z' }

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
  },
  {
    name: 'failed update names the stop reason',
    receipt: { ...updateBase, stop_reason: 'RuntimeError: gateway never came back' },
    headline: /Last update failed.*gateway never came back/,
    tone: 'text-destructive'
  },
  {
    name: 'failed update falls back to the failed step',
    receipt: {
      ...updateBase,
      stop_reason: null,
      steps: [{ name: 'venv_sync', ok: false, detail: 'uv sync exited 1' }]
    },
    headline: /Last update failed.*uv sync exited 1/,
    tone: 'text-destructive'
  },
  {
    name: 'successful update is surfaced, not buried',
    receipt: { ...updateBase, outcome: 'success' },
    headline: /Last update succeeded/,
    tone: 'text-muted-foreground'
  },
  {
    name: 'interrupted update points at rerun',
    receipt: { ...updateBase, outcome: 'interrupted', stop_reason: null },
    headline: /was interrupted.*rerun `hermes update`/,
    tone: 'text-amber-600'
  },
  {
    name: 'running update shows the live tail message',
    receipt: { ...updateBase, outcome: 'running', stop_reason: null },
    tail: freshTail,
    headline: /Update in progress — Building update products/,
    tone: 'text-muted-foreground'
  },
  {
    name: 'stale tail reads as wedged, not working',
    receipt: { ...updateBase, outcome: 'running', stop_reason: null },
    tail: staleTail,
    headline: /Update in progress — no progress for/,
    tone: 'text-amber-600'
  },
  {
    name: 'running update without a tail falls back to the last step',
    receipt: { ...updateBase, outcome: 'running', stop_reason: null },
    tail: null,
    headline: /Update in progress — sibling_profile_snapshots/,
    tone: 'text-muted-foreground'
  }
]

it.each(cases)(
  '$name receipt paints the dominant warning and subordinate rows',
  async ({ receipt, tail, headline, tone }: ReceiptCase): Promise<void> => {
    vi.stubGlobal('hermesDesktop', {
      getSyncStatus: async (): Promise<DesktopSyncReceipt | null> => receipt,
      getUpdateStage: async () => tail ?? null
    } satisfies Pick<Window['hermesDesktop'], 'getSyncStatus' | 'getUpdateStage'>)
    await act(async (): Promise<void> => {
      render(<SyncStatusCard />)
    })

    if (!headline) {
      expect(screen.queryByTestId('sync-status-card')).toBeNull()

      return
    }

    expect(screen.getByText(headline).className).toContain(tone)

    if (receipt && 'plugin_bisect' in receipt && receipt.plugin_bisect) {
      expect(screen.getByText('conflictor: conflicts with core pin')).toBeTruthy()
    }

    if (receipt?.plugin_checks?.some(row => row.needs_fixing)) {
      expect(screen.getByText('bad-url: update_url points at a fork')).toBeTruthy()
    }

    if (receipt?.plugin_checks?.some(row => row.update_available === true)) {
      expect(screen.getByText('grower: 1.0.0 → 1.1.0')).toBeTruthy()
      expect(screen.queryByText('fine')).toBeNull()
      expect(screen.queryByText('kept')).toBeNull()
    }
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

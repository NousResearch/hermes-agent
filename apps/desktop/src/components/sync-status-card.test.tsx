import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { DesktopSyncReceipt } from '@/global'

import { SyncStatusCard } from './sync-status-card'

function setBridge(receipt: DesktopSyncReceipt | null | Promise<DesktopSyncReceipt | null>, failing = false) {
  ;(window as any).hermesDesktop = {
    getSyncStatus: failing
      ? vi.fn(() => Promise.reject(new Error('bridge gone')))
      : vi.fn(() => Promise.resolve(receipt))
  }
}

afterEach(() => {
  cleanup()
  delete (window as any).hermesDesktop
})

async function renderCard() {
  await act(async () => {
    render(<SyncStatusCard />)
  })
}

describe("SyncStatusCard — the renderer's getSyncStatus warning surface", () => {
  it('renders the needs-fixing warning headline and per-plugin reasons', async () => {
    setBridge({
      kind: 'sync',
      outcome: 'checked',
      plugin_checks: [
        { name: 'bad-url', needs_fixing: 'update_url points at a fork', update_available: false },
        { name: 'fine', update_available: false }
      ]
    })
    await renderCard()

    expect(screen.getByTestId('sync-status-card')).toBeTruthy()
    expect(screen.getByText(/need update-url review/)).toBeTruthy()
    expect(screen.getByText(/bad-url: update_url points at a fork/)).toBeTruthy()
    expect(screen.queryByText(/fine/)).toBeNull()
  })

  it('renders bisect disables and available updates', async () => {
    setBridge({
      kind: 'sync',
      outcome: 'bisected',
      plugin_bisect: [{ plugin: 'conflictor', action: 'disabled', reason: 'conflicts with core pin' }],
      plugin_checks: [{ name: 'grower', update_available: true, current: '1.0.0', latest: '1.1.0' }]
    })
    await renderCard()

    expect(screen.getByText(/disabled by dependency conflicts/)).toBeTruthy()
    expect(screen.getByText(/conflictor: conflicts with core pin/)).toBeTruthy()
    expect(screen.getByText(/grower: 1.0.0 → 1.1.0/)).toBeTruthy()
  })

  it('renders a failed venv rebuild as the top signal', async () => {
    setBridge({
      kind: 'sync',
      outcome: 'failed',
      venv_rebuild: { ok: false, reason: 'uv sync exited 1' }
    })
    await renderCard()

    expect(screen.getByText(/rebuild failed/)).toBeTruthy()
  })

  it('renders nothing without a receipt and nothing on a healthy receipt', async () => {
    setBridge(null)
    const { container } = render(<SyncStatusCard />)
    await act(async () => {})
    expect(container.textContent).toBe('')

    cleanup()
    setBridge({ kind: 'sync', outcome: 'ok' })
    await renderCard()
    expect(screen.queryByTestId('sync-status-card')).toBeNull()
  })

  it('a broken bridge degrades to nothing instead of breaking the updates overlay', async () => {
    setBridge(null, true)
    const { container } = render(<SyncStatusCard />)
    await act(async () => {})
    expect(container.textContent).toBe('')
  })
})

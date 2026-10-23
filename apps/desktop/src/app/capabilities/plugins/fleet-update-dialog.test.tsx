// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { FleetInstall } from '@/store/plugin-fleet-updates'

const confirmMock = vi.fn()
const updateMock = vi.fn()

vi.mock('@/store/confirm', () => ({ confirm: (...args: unknown[]) => confirmMock(...args) }))

vi.mock('@/store/plugin-fleet-updates', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  settleFleetInstalls: vi.fn(),
  updateFleetInstall: (...args: unknown[]) => updateMock(...args)
}))

const { FleetUpdateDialog } = await import('./fleet-update-dialog')

const install = (connectionId: string, label: string, profile: string): FleetInstall => ({
  target: { connectionId, connectionLabel: label, profile, targetProfile: profile, current: false, via: 'default' },
  row: {
    name: 'claude-sub',
    version: '0.3.0',
    description: '',
    source: 'user',
    status: 'enabled',
    catalog_version: '0.3.3'
  }
})

const installs = [
  install('local', 'This device', 'default'),
  install('local', 'This device', 'coder'),
  install('box', 'Box', 'work')
]

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

async function runDialog(onFinished = vi.fn()) {
  render(
    <FleetUpdateDialog installs={installs} onClose={vi.fn()} onFinished={onFinished} open skipped={[]} title="Update" />
  )
  await act(async () => {
    fireEvent.click(screen.getByRole('button', { name: 'Update 3' }))
  })
  await screen.findByRole('button', { name: 'Done' })

  return onFinished
}

describe('FleetUpdateDialog', () => {
  it('groups rows under their gateway', () => {
    render(<FleetUpdateDialog installs={installs} onClose={vi.fn()} open skipped={[]} title="Update" />)

    expect(screen.getAllByRole('group').map(group => group.textContent?.split('default')[0].split('work')[0])).toEqual(
      expect.arrayContaining([expect.stringContaining('This device'), expect.stringContaining('Box')])
    )
  })

  it('asks for widened permissions once for the whole batch, and only counts what was applied', async () => {
    // default needs consent, coder applies, work needs consent.
    updateMock.mockImplementation(async (target: FleetInstall, options: { acceptCapabilities?: boolean }) =>
      target.target.profile === 'coder' || options?.acceptCapabilities
        ? { kind: 'applied' }
        : { kind: 'consent', sha: 'cccccccc', deltaLines: ['+ tool x'] }
    )
    confirmMock.mockResolvedValue(false)

    const onFinished = await runDialog()

    expect(confirmMock).toHaveBeenCalledTimes(1)
    expect(confirmMock.mock.calls[0][0].description).toContain('2 profiles')
    expect(onFinished).toHaveBeenCalledWith(1)
    expect(screen.getByText('1 updated. 2 not applied because the new permissions were declined.')).toBeTruthy()
    expect(screen.getAllByText('Not applied')).toHaveLength(2)
  })

  it('asks once to override a caution scan for the batch, and retries only those rows with force', async () => {
    const finding = { category: 'execution', description: 'eval', file: 'helper.py', line: 1, severity: 'high' }

    // default and work get a caution verdict; coder applies straight away.
    updateMock.mockImplementation(async (target: FleetInstall, options: { force?: boolean }) =>
      target.target.profile === 'coder' || options?.force
        ? { kind: 'applied' }
        : { kind: 'scan', verdict: 'caution', findings: [finding] }
    )
    confirmMock.mockResolvedValue(true)

    const onFinished = await runDialog()

    expect(confirmMock).toHaveBeenCalledTimes(1)
    expect(confirmMock.mock.calls[0][0]).toMatchObject({ destructive: true, confirmLabel: 'Update anyway' })
    // The same finding from two profiles is listed once.
    expect(confirmMock.mock.calls[0][0].description.match(/helper\.py:1/g)).toHaveLength(1)
    expect(updateMock.mock.calls.filter(call => call[1]?.force)).toHaveLength(2)
    expect(onFinished).toHaveBeenCalledWith(3)
  })

  it('fails a dangerous scan verdict without offering an override', async () => {
    updateMock.mockImplementation(async (target: FleetInstall) =>
      target.target.profile === 'work' ? { kind: 'scan', verdict: 'dangerous', findings: [] } : { kind: 'applied' }
    )

    await runDialog()

    expect(confirmMock).not.toHaveBeenCalled()
    await waitFor(() => expect(screen.getByText(/2 updated, 1 failed/)).toBeTruthy())
  })

  it('reports failures with the real success count', async () => {
    updateMock.mockImplementation(async (target: FleetInstall) =>
      target.target.profile === 'work' ? { kind: 'failed', error: 'socket closed' } : { kind: 'applied' }
    )

    await runDialog()

    await waitFor(() => expect(screen.getByText(/2 updated, 1 failed/)).toBeTruthy())
    expect(confirmMock).not.toHaveBeenCalled()
  })
})

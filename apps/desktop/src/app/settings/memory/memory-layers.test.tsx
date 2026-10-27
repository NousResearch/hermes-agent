import { cleanup, render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import { getMemoryStatus } from '@/hermes'
import type { MemoryStatusResponse } from '@/types/hermes'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  getCuratorStatus: vi.fn(() => new Promise(() => {})),
  getMemoryStatus: vi.fn()
}))

const { MaintenancePanel } = await import('../../command-center/maintenance')
const { MemoryLayersPanel } = await import('./memory-layers')

function status(active: string): MemoryStatusResponse {
  return {
    active,
    providers: [
      { name: 'byterover', description: '', available: true, configured: true, status: 'ready' },
      { name: 'retaindb', description: '', available: false, configured: false, status: 'needs_config' }
    ],
    builtin_files: { memory: 67, user: 17 }
  }
}

afterEach(cleanup)

// #145175: an active external provider read as "not in effect" because the
// Memory surfaces showed only MEMORY.md / USER.md.
describe('memory layers', () => {
  it('names the active provider and its readiness, and marks the built-in layer always on', async () => {
    vi.mocked(getMemoryStatus).mockResolvedValue(status('byterover'))
    render(<MaintenancePanel />)

    const layers = await screen.findByTestId('memory-layers')
    expect(within(layers).getByText('Active memory provider')).toBeTruthy()
    expect(within(layers).getByText('byterover')).toBeTruthy()
    expect(within(layers).getByText('ready')).toBeTruthy()
    expect(within(layers).getByText('Built-in layer')).toBeTruthy()
    expect(within(layers).getByText('Always on')).toBeTruthy()
  })

  it('says built-in only when no external provider is configured, and keeps the layer always on', async () => {
    vi.mocked(getMemoryStatus).mockResolvedValue(status(''))
    render(<MaintenancePanel />)

    const layers = await screen.findByTestId('memory-layers')
    expect(within(layers).getByText('None — built-in memory only')).toBeTruthy()
    expect(within(layers).queryByText('byterover')).toBeNull()
    expect(within(layers).getByText('Always on')).toBeTruthy()
  })

  it('follows the Settings page draft provider over the last saved one', async () => {
    vi.mocked(getMemoryStatus).mockResolvedValue(status('byterover'))
    render(<MemoryLayersPanel provider="retaindb" />)

    const layers = await screen.findByTestId('memory-layers')
    expect(await within(layers).findByText('needs setup')).toBeTruthy()
    expect(within(layers).getByText('retaindb')).toBeTruthy()
    expect(within(layers).queryByText('byterover')).toBeNull()
  })
})

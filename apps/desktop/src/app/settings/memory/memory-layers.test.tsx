import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import { getMemoryProviderConfig, getMemoryStatus, saveMemoryProviderConfig } from '@/hermes'
import type { MemoryProviderConfig, MemoryProviderStatus, MemoryStatusResponse } from '@/types/hermes'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  getCuratorStatus: vi.fn(() => new Promise(() => {})),
  getMemoryProviderConfig: vi.fn(),
  getMemoryStatus: vi.fn(),
  saveMemoryProviderConfig: vi.fn()
}))

const { MaintenancePanel } = await import('../../command-center/maintenance')
const { MemoryLayersPanel, MemoryLayersSummary } = await import('./memory-layers')
const { ProviderConfigPanel } = await import('./provider-config-panel')

function status(active: string, readiness: Partial<Record<string, MemoryProviderStatus>> = {}): MemoryStatusResponse {
  return {
    active,
    providers: [
      { name: 'byterover', description: '', available: true, configured: true, status: readiness.byterover ?? 'ready' },
      {
        name: 'retaindb',
        description: '',
        available: false,
        configured: false,
        status: readiness.retaindb ?? 'needs_config'
      }
    ],
    builtin_files: { memory: 67, user: 17 }
  }
}

function retaindbConfig(): MemoryProviderConfig {
  return {
    name: 'retaindb',
    label: 'RetainDB',
    docs_url: '',
    fields: [
      {
        key: 'apiKey',
        label: 'API key',
        kind: 'secret',
        value: '',
        description: '',
        placeholder: 'Enter RetainDB API key',
        is_set: false,
        inline: true,
        group: 'Connection',
        options: []
      }
    ]
  }
}

// The ListRow whose title is `title`, so assertions read one row, not the card.
function row(title: string): HTMLElement {
  const layers = screen.getByTestId('memory-layers')
  const label = within(layers).getByText(title)
  const container = [...layers.children].find(child => child.contains(label))

  if (!(container instanceof HTMLElement)) {
    throw new Error(`no row titled ${title}`)
  }

  return container
}

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

// #145175: an active external provider read as "not in effect" because the
// Memory surfaces showed only MEMORY.md / USER.md.
describe('memory layers', () => {
  it('names the active provider and its readiness, and marks the built-in layer on by default', async () => {
    vi.mocked(getMemoryStatus).mockResolvedValue(status('byterover'))
    render(<MaintenancePanel />)

    await screen.findByTestId('memory-layers')
    expect(within(row('Active memory provider')).getByText('byterover')).toBeTruthy()
    expect(within(row('Active memory provider')).getByText('ready')).toBeTruthy()
    expect(within(row('Built-in layer')).getByText('On by default')).toBeTruthy()
  })

  it('says built-in only when no external provider is configured', async () => {
    vi.mocked(getMemoryStatus).mockResolvedValue(status(''))
    render(<MaintenancePanel />)

    const layers = await screen.findByTestId('memory-layers')
    expect(within(layers).getByText('None — built-in memory only')).toBeTruthy()
    expect(within(layers).queryByText('byterover')).toBeNull()
  })

  // Review on #145574: the unsaved Settings draft was labelled active even
  // when GET /api/memory named a different provider.
  it('calls an unsaved draft "selected", keeps the saved one active, and flips once the save refetches', async () => {
    vi.mocked(getMemoryStatus).mockResolvedValue(status('byterover', { retaindb: 'ready' }))
    render(<MemoryLayersPanel provider="retaindb" />)

    await screen.findByText('Active memory provider')
    expect(within(row('Selected memory provider')).getByText('retaindb')).toBeTruthy()
    expect(within(row('Selected memory provider')).getByText(/Not active yet/)).toBeTruthy()
    expect(within(row('Active memory provider')).getByText('byterover')).toBeTruthy()
    expect(within(row('Active memory provider')).queryByText('retaindb')).toBeNull()

    const { refreshMemoryStatus } = await import('@/store/memory-status')
    vi.mocked(getMemoryStatus).mockResolvedValue(status('retaindb', { retaindb: 'ready' }))
    act(() => refreshMemoryStatus())

    await waitFor(() => expect(screen.queryByText('Selected memory provider')).toBeNull())
    expect(within(row('Active memory provider')).getByText('retaindb')).toBeTruthy()
  })

  // Review on #145574: readiness stayed "needs setup" after a successful
  // credential save, and a failed GET dropped the badge silently.
  it('refreshes readiness after a provider credential save and offers a retry when the status read fails', async () => {
    vi.mocked(getMemoryStatus).mockRejectedValueOnce(new Error('offline'))
    vi.mocked(getMemoryProviderConfig).mockResolvedValue(retaindbConfig())
    vi.mocked(saveMemoryProviderConfig).mockResolvedValue({ ok: true } as never)
    render(
      <>
        <MemoryLayersPanel provider="retaindb" />
        <ProviderConfigPanel provider="retaindb" />
      </>
    )

    await screen.findByText('Memory status unavailable')
    vi.mocked(getMemoryStatus).mockResolvedValue(status('retaindb'))
    fireEvent.click(within(row('Memory status unavailable')).getByRole('button', { name: 'Retry' }))
    expect(await within(await waitFor(() => row('Active memory provider'))).findByText('needs setup')).toBeTruthy()
    expect(screen.queryByText('Memory status unavailable')).toBeNull()

    vi.mocked(getMemoryStatus).mockResolvedValue(status('retaindb', { retaindb: 'ready' }))
    const apiKey = await screen.findByPlaceholderText('Enter RetainDB API key')
    fireEvent.change(apiKey, { target: { value: 'fixture-key' } })
    fireEvent.blur(apiKey)

    await waitFor(() => expect(saveMemoryProviderConfig).toHaveBeenCalled())
    expect(await within(row('Active memory provider')).findByText('ready')).toBeTruthy()
  })

  // Review on #145574: the copy said both files load into every session and
  // that a not-ready provider recalls.
  it('reflects the built-in toggles and claims recall only for a ready provider', () => {
    render(<MemoryLayersSummary builtin={{ memory: false, user: true }} status={status('retaindb')} />)

    expect(within(row('Built-in layer')).getByText('USER.md only')).toBeTruthy()
    expect(within(row('Active memory provider')).queryByText(/adds its own recall/)).toBeNull()
    expect(within(row('Active memory provider')).getByText(/no recall until it is ready/)).toBeTruthy()
  })

  // Review on #145574: `Default` is a backend built-in sentinel but rendered
  // as an external provider.
  it('treats a saved "Default" provider as built-in memory', async () => {
    vi.mocked(getMemoryStatus).mockResolvedValue(status(''))
    render(<MemoryLayersPanel provider="Default" />)

    await screen.findByText('Active memory provider')
    expect(screen.getByText('None — built-in memory only')).toBeTruthy()
    expect(screen.queryByText('Default')).toBeNull()
    expect(screen.queryByText('Selected memory provider')).toBeNull()
  })
})

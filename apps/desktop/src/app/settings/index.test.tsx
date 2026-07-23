import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { useEffect } from 'react'
import { MemoryRouter, useLocation } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import type { EnvVarInfo } from '@/types/hermes'

const getEnvVars = vi.fn()
const getHermesConfigRecord = vi.fn()
const getHermesConfigSchema = vi.fn()
const configUnmounted = vi.fn()

class TestResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

vi.stubGlobal('ResizeObserver', TestResizeObserver)

vi.mock('@/hermes', async importOriginal => {
  const actual = await importOriginal<typeof HermesApi>()

  return {
    ...actual,
    getEnvVars: () => getEnvVars(),
    getHermesConfigRecord: () => getHermesConfigRecord(),
    getHermesConfigSchema: () => getHermesConfigSchema()
  }
})

vi.mock('./config-settings', () => ({
  ConfigSettings: ({ activeSectionId }: { activeSectionId: string }) => {
    useEffect(() => () => configUnmounted(), [])

    return <div>Config page: {activeSectionId}</div>
  }
}))

function envVar(category: string, patch: Partial<EnvVarInfo> = {}): EnvVarInfo {
  return {
    advanced: false,
    category,
    description: '',
    is_password: true,
    is_set: false,
    redacted_value: null,
    tools: [],
    url: '',
    ...patch
  }
}

function LocationProbe() {
  const location = useLocation()

  return <output data-testid="location">{`${location.pathname}${location.search}`}</output>
}

function searchResultsFor(search: HTMLElement) {
  const listId = search.getAttribute('aria-controls')
  const list = listId ? globalThis.document.getElementById(listId) : null

  if (!list) {
    throw new Error('Search input is not connected to its result list')
  }

  return within(list)
}

async function renderSettings() {
  const { SettingsView } = await import('./index')
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  await act(async () => {
    render(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={['/settings?tab=config:model']}>
          <SettingsView onClose={vi.fn()} />
          <LocationProbe />
        </MemoryRouter>
      </QueryClientProvider>
    )
  })
}

beforeEach(() => {
  Object.defineProperty(Element.prototype, 'scrollIntoView', {
    configurable: true,
    value: vi.fn()
  })
  getHermesConfigRecord.mockResolvedValue({ approvals: { timeout: 120 } })
  getHermesConfigSchema.mockResolvedValue({
    fields: {
      'approvals.timeout': {
        description: 'Seconds to wait for approval.',
        type: 'number'
      }
    }
  })
  getEnvVars.mockResolvedValue({
    FUTURE_CRAWLER_API_KEY: envVar('tool', {
      description: 'Fetch structured pages from a new crawler.',
      tools: ['future_crawl']
    })
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('SettingsView global search', () => {
  it('finds config fields and dynamic credentials, then opens their exact targets', async () => {
    await renderSettings()

    const [search] = await screen.findAllByRole('combobox', { name: 'Search all settings…' })

    fireEvent.change(search, { target: { value: 'approval timeout' } })
    expect(await searchResultsFor(search).findByRole('option', { name: /Approval Timeout.*Safety/ })).toBeTruthy()
    fireEvent.keyDown(search, { key: 'Enter' })

    await waitFor(() => {
      expect(screen.getByTestId('location').textContent).toContain('tab=config%3Asafety')
      expect(screen.getByTestId('location').textContent).toContain('field=approvals.timeout')
    })
    expect((search as HTMLInputElement).value).toBe('')
    expect(screen.getByText('Config page: safety')).toBeTruthy()

    fireEvent.change(search, { target: { value: 'structured crawler' } })
    const credentialResult = await searchResultsFor(search).findByRole('option', { name: /FUTURE CRAWLER.*Tools/ })
    fireEvent.click(credentialResult)

    await waitFor(() => {
      expect(screen.getByTestId('location').textContent).toContain('tab=keys')
      expect(screen.getByTestId('location').textContent).toContain('kview=tools')
      expect(screen.getByTestId('location').textContent).toContain('key=FUTURE_CRAWLER_API_KEY')
    })
    expect((search as HTMLInputElement).value).toBe('')
  })

  it('does not announce an empty result while the catalog is loading', async () => {
    getEnvVars.mockReturnValue(new Promise(() => {}))

    await renderSettings()

    const [search] = await screen.findAllByRole('combobox', { name: 'Search all settings…' })
    fireEvent.change(search, { target: { value: 'future crawler' } })

    expect((await screen.findAllByText('Searching settings…')).length).toBeGreaterThan(0)
    expect(screen.queryByText('No settings match your search.')).toBeNull()
  })

  it('keeps the active settings page mounted while the user searches', async () => {
    await renderSettings()

    const [search] = await screen.findAllByRole('combobox', { name: 'Search all settings…' })
    fireEvent.change(search, { target: { value: 'theme' } })

    expect(await searchResultsFor(search).findByRole('option', { name: /Theme.*Appearance/ })).toBeTruthy()
    expect(configUnmounted).not.toHaveBeenCalled()

    fireEvent.change(search, { target: { value: '' } })
    expect(screen.getByText('Config page: model')).toBeTruthy()
    expect(configUnmounted).not.toHaveBeenCalled()
  })

  it('keeps page destinations searchable for bespoke settings surfaces', async () => {
    await renderSettings()

    const [search] = await screen.findAllByRole('combobox', { name: 'Search all settings…' })
    fireEvent.change(search, { target: { value: 'theme' } })

    const appearanceResult = await searchResultsFor(search).findByRole('option', { name: /Theme.*Appearance/ })
    fireEvent.click(appearanceResult)

    await waitFor(() => {
      const target = globalThis.document.getElementById('setting-field-appearance.theme')
      expect(target?.classList).toContain('setting-field-highlight')
      expect(globalThis.document.activeElement).toBe(target)
    })
    await waitFor(() => {
      expect(screen.getByTestId('location').textContent).toContain('tab=config%3Aappearance')
      expect(screen.getByTestId('location').textContent).not.toContain('setting=')
    })
  })
})

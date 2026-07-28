import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, useNavigate } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { EnvVarInfo } from '@/types/hermes'

const getEnvVars = vi.fn()

class TestResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

vi.stubGlobal('ResizeObserver', TestResizeObserver)

vi.mock('@/hermes', () => ({
  deleteEnvVar: vi.fn(),
  getEnvVars: () => getEnvVars(),
  revealEnvVar: vi.fn(),
  setEnvVar: vi.fn()
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

beforeEach(() => {
  getEnvVars.mockResolvedValue({})
  Object.defineProperty(Element.prototype, 'scrollIntoView', {
    configurable: true,
    value: vi.fn()
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

async function renderKeysSettings(view: 'settings' | 'tools', route = '/settings') {
  const { KeysSettings } = await import('./keys-settings')

  await act(async () => {
    render(
      <MemoryRouter initialEntries={[route]}>
        <KeysSettings view={view} />
      </MemoryRouter>
    )
  })
}

function DeepLinkButton({ target }: { target: string }) {
  const navigate = useNavigate()

  return (
    <button onClick={() => navigate(`/settings?tab=keys&key=${target}`)} type="button">
      Open key
    </button>
  )
}

describe('KeysSettings scoped command search', () => {
  it('curates tool results by label, description, env key, URL, and tool name', async () => {
    getEnvVars.mockResolvedValue({
      BRAVE_SEARCH_API_KEY: envVar('tool', {
        description: 'Search the web with Brave.',
        url: 'https://brave.com/search/api/'
      }),
      FIRECRAWL_API_KEY: envVar('tool', {
        description: 'Crawl and extract websites.',
        tools: ['browser_navigate']
      }),
      GATEWAY_PROXY: envVar('setting', { description: 'Gateway reverse proxy.' })
    })

    await renderKeysSettings('tools')

    const search = await screen.findByRole('combobox', { name: 'Search tools…' })
    expect(screen.getByText('BRAVE SEARCH')).toBeTruthy()
    expect(screen.getByText('FIRECRAWL')).toBeTruthy()
    expect(screen.queryByText('GATEWAY PROXY')).toBeNull()
    expect(screen.queryByRole('option')).toBeNull()
    expect(search.getAttribute('aria-expanded')).toBe('false')

    fireEvent.change(search, { target: { value: 'crawl and extract' } })
    const firecrawlResult = await screen.findByRole('option', { name: /FIRECRAWL/ })
    expect(firecrawlResult.getAttribute('data-selected')).toBe('true')
    expect(screen.queryByRole('option', { name: /BRAVE SEARCH/ })).toBeNull()

    fireEvent.change(search, { target: { value: 'brave_search_api' } })
    expect(await screen.findByRole('option', { name: /BRAVE SEARCH/ })).toBeTruthy()
    expect(screen.queryByRole('option', { name: /FIRECRAWL/ })).toBeNull()

    fireEvent.change(search, { target: { value: 'brave.com' } })
    expect(await screen.findByRole('option', { name: /BRAVE SEARCH/ })).toBeTruthy()

    fireEvent.change(search, { target: { value: 'browser_navigate' } })
    expect(await screen.findByRole('option', { name: /FIRECRAWL/ })).toBeTruthy()
    expect(screen.queryByRole('option', { name: /BRAVE SEARCH/ })).toBeNull()

    fireEvent.change(search, { target: { value: 'does-not-exist' } })
    expect(await screen.findByText('No entries match your search.')).toBeTruthy()
  })

  it('scopes settings results and excludes channel-managed credentials', async () => {
    getEnvVars.mockResolvedValue({
      API_SERVER_TOKEN: envVar('setting', { description: 'Protect the local API server.' }),
      GATEWAY_PROXY: envVar('messaging', { description: 'Gateway reverse proxy address.' }),
      TELEGRAM_BOT_TOKEN: envVar('messaging', {
        channel_managed: true,
        description: 'Telegram bot token.'
      }),
      BRAVE_SEARCH_API_KEY: envVar('tool', { description: 'Search the web with Brave.' })
    })

    await renderKeysSettings('settings')

    const search = await screen.findByRole('combobox', { name: 'Search settings…' })
    expect(screen.getByText('API SERVER')).toBeTruthy()
    expect(screen.getByText('GATEWAY PROXY')).toBeTruthy()
    expect(screen.queryByText('TELEGRAM BOT')).toBeNull()
    expect(screen.queryByText('BRAVE SEARCH')).toBeNull()

    fireEvent.change(search, { target: { value: 'reverse proxy' } })
    expect(await screen.findByRole('option', { name: /GATEWAY PROXY/ })).toBeTruthy()
    expect(screen.queryByRole('option', { name: /API SERVER/ })).toBeNull()
  })

  it('selects a command result, then expands and highlights its credential card', async () => {
    getEnvVars.mockResolvedValue({
      BRAVE_SEARCH_API_KEY: envVar('tool', { description: 'Search the web with Brave.' }),
      FIRECRAWL_API_KEY: envVar('tool', {
        description: 'Crawl and extract websites.',
        tools: ['browser_navigate']
      })
    })

    await renderKeysSettings('tools', '/settings?tab=keys&kview=tools')

    const search = await screen.findByRole('combobox', { name: 'Search tools…' })
    fireEvent.change(search, { target: { value: 'browser_navigate' } })
    await screen.findByRole('option', { name: /FIRECRAWL/ })
    fireEvent.keyDown(search, { key: 'Enter' })

    await waitFor(() => expect((search as HTMLInputElement).value).toBe(''))
    await waitFor(() => {
      const target = globalThis.document.getElementById('credential-key-FIRECRAWL_API_KEY')
      expect(target?.classList).toContain('setting-field-highlight')
    })
    expect(screen.getByText('Crawl and extract websites.')).toBeTruthy()
  })

  it('dismisses outside and reopens on focus while Escape clears the query', async () => {
    getEnvVars.mockResolvedValue({
      BRAVE_SEARCH_API_KEY: envVar('tool', { description: 'Search the web with Brave.' })
    })

    await renderKeysSettings('tools')

    const search = await screen.findByRole('combobox', { name: 'Search tools…' })
    fireEvent.change(search, { target: { value: 'brave' } })
    expect(await screen.findByRole('option', { name: /BRAVE SEARCH/ })).toBeTruthy()
    expect(search.getAttribute('aria-expanded')).toBe('true')

    fireEvent.pointerDown(globalThis.document.body)
    await waitFor(() => expect(search.getAttribute('aria-expanded')).toBe('false'))
    expect((search as HTMLInputElement).value).toBe('brave')

    fireEvent.focus(search)
    await waitFor(() => expect(search.getAttribute('aria-expanded')).toBe('true'))
    fireEvent.keyDown(search, { key: 'Escape' })
    expect((search as HTMLInputElement).value).toBe('')
    expect(search.getAttribute('aria-expanded')).toBe('false')
  })

  it('clears the scoped query when a deep link opens a key in the active view', async () => {
    getEnvVars.mockResolvedValue({
      BRAVE_SEARCH_API_KEY: envVar('tool', { description: 'Search the web with Brave.' }),
      FIRECRAWL_API_KEY: envVar('tool', { description: 'Crawl and extract websites.' })
    })
    const { KeysSettings } = await import('./keys-settings')

    render(
      <MemoryRouter initialEntries={['/settings?tab=keys']}>
        <KeysSettings view="tools" />
        <DeepLinkButton target="BRAVE_SEARCH_API_KEY" />
      </MemoryRouter>
    )

    const search = await screen.findByRole('combobox', { name: 'Search tools…' })
    fireEvent.change(search, { target: { value: 'firecrawl' } })
    expect(await screen.findByRole('option', { name: /FIRECRAWL/ })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Open key' }))

    await waitFor(() => expect((search as HTMLInputElement).value).toBe(''))
    expect(await screen.findByText('Search the web with Brave.')).toBeTruthy()
  })

  it('does not clear a query for a deep link owned by the other sub-view', async () => {
    getEnvVars.mockResolvedValue({
      BRAVE_SEARCH_API_KEY: envVar('tool', { description: 'Search the web with Brave.' }),
      GATEWAY_PROXY: envVar('setting', { description: 'Gateway reverse proxy address.' })
    })

    await renderKeysSettings('settings', '/settings?tab=keys&kview=settings&key=BRAVE_SEARCH_API_KEY')

    const search = await screen.findByRole('combobox', { name: 'Search settings…' })
    fireEvent.change(search, { target: { value: 'reverse proxy' } })

    expect((search as HTMLInputElement).value).toBe('reverse proxy')
    expect(await screen.findByRole('option', { name: /GATEWAY PROXY/ })).toBeTruthy()
  })
})

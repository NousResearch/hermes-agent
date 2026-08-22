import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, useNavigate } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { stubResizeObserver } from '@/test/jsdom'

import { envVar } from './test-utils'

const getEnvVars = vi.fn()

stubResizeObserver()

vi.mock('@/hermes', () => ({
  deleteEnvVar: vi.fn(),
  getEnvVars: () => getEnvVars(),
  revealEnvVar: vi.fn(),
  setApiRequestProfile: () => undefined,
  setEnvVar: vi.fn()
}))

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

describe('KeysSettings', () => {
  it('lists tools and excludes settings / channel-managed credentials', async () => {
    getEnvVars.mockResolvedValue({
      BRAVE_SEARCH_API_KEY: envVar('tool', { description: 'Search the web with Brave.' }),
      FIRECRAWL_API_KEY: envVar('tool', { description: 'Crawl and extract websites.' }),
      GATEWAY_PROXY: envVar('setting', { description: 'Gateway reverse proxy.' }),
      TELEGRAM_BOT_TOKEN: envVar('messaging', {
        channel_managed: true,
        description: 'Telegram bot token.'
      })
    })

    await renderKeysSettings('tools')

    expect(screen.getByText('BRAVE SEARCH')).toBeTruthy()
    expect(screen.getByText('FIRECRAWL')).toBeTruthy()
    expect(screen.queryByText('GATEWAY PROXY')).toBeNull()
    expect(screen.queryByText('TELEGRAM BOT')).toBeNull()
    expect(screen.queryByRole('combobox')).toBeNull()
  })

  it('lists settings rows and excludes tools / channel-managed credentials', async () => {
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

    expect(screen.getByText('API SERVER')).toBeTruthy()
    expect(screen.getByText('GATEWAY PROXY')).toBeTruthy()
    expect(screen.queryByText('TELEGRAM BOT')).toBeNull()
    expect(screen.queryByText('BRAVE SEARCH')).toBeNull()
  })

  it('expands and highlights a deep-linked credential card', async () => {
    getEnvVars.mockResolvedValue({
      BRAVE_SEARCH_API_KEY: envVar('tool', { description: 'Search the web with Brave.' }),
      FIRECRAWL_API_KEY: envVar('tool', { description: 'Crawl and extract websites.' })
    })

    const { KeysSettings } = await import('./keys-settings')

    render(
      <MemoryRouter initialEntries={['/settings?tab=keys']}>
        <KeysSettings view="tools" />
        <DeepLinkButton target="FIRECRAWL_API_KEY" />
      </MemoryRouter>
    )

    expect(await screen.findByText('BRAVE SEARCH')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Open key' }))

    await waitFor(() => {
      const target = globalThis.document.getElementById('credential-key-FIRECRAWL_API_KEY')
      expect(target?.classList).toContain('setting-field-highlight')
    })
    expect(screen.getByText('Crawl and extract websites.')).toBeTruthy()
  })
})

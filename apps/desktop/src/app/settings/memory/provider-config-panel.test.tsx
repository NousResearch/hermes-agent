import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { MemoryProviderConfig } from '@/types/hermes'

const getMemoryProviderConfig = vi.fn()
const saveMemoryProviderConfig = vi.fn()

vi.mock('@/hermes', () => ({
  getMemoryProviderConfig: (provider: string) => getMemoryProviderConfig(provider),
  saveMemoryProviderConfig: (provider: string, values: unknown) => saveMemoryProviderConfig(provider, values)
}))

vi.mock('@/store/profile', async () => {
  const { atom } = await import('nanostores')
  return { $activeGatewayProfile: atom('default') }
})

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

function honchoSchema(): MemoryProviderConfig {
  return {
    name: 'honcho',
    label: 'Honcho',
    docs_url: 'https://docs.honcho.dev/v3/guides/integrations/hermes',
    fields: [
      {
        key: 'apiKey',
        label: 'API key',
        kind: 'secret',
        value: '',
        description: 'Authenticate with Honcho Cloud.',
        placeholder: 'Enter Honcho API key',
        is_set: false,
        inline: true,
        group: 'Connection',
        options: []
      },
      {
        key: 'baseUrl',
        label: 'Base URL',
        kind: 'text',
        value: '',
        description: 'Self-hosted Honcho URL.',
        placeholder: 'https://… (self-hosted)',
        is_set: false,
        inline: true,
        group: 'Connection',
        options: []
      },
      {
        key: 'environment',
        label: 'Environment',
        kind: 'select',
        value: 'production',
        description: 'Honcho environment.',
        placeholder: '',
        is_set: true,
        inline: true,
        group: 'Connection',
        options: [
          { value: 'production', label: 'Production', description: '' },
          { value: 'demo', label: 'Demo', description: '' },
          { value: 'local', label: 'Local', description: '' }
        ]
      },
      {
        key: 'workspace',
        label: 'Workspace',
        kind: 'text',
        value: 'myws',
        description: 'Honcho workspace ID.',
        placeholder: 'hermes',
        is_set: true,
        inline: true,
        group: 'Connection',
        options: []
      },
      // Non-inline field: must NOT render in the compact panel and must NOT be
      // submitted when the panel saves.
      {
        key: 'writeFrequency',
        label: 'Write frequency',
        kind: 'text',
        value: 'async',
        description: '',
        placeholder: '',
        is_set: true,
        inline: false,
        group: 'Message writing',
        options: []
      }
    ]
  }
}

beforeEach(() => {
  getMemoryProviderConfig.mockResolvedValue(honchoSchema())
  saveMemoryProviderConfig.mockResolvedValue({ ok: true })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

async function renderPanel(provider = 'honcho') {
  const { ProviderConfigPanel } = await import('./provider-config-panel')

  return render(<ProviderConfigPanel provider={provider} />)
}

describe('ProviderConfigPanel', () => {
  it('renders the declared inline fields generically', async () => {
    await renderPanel()

    expect(await screen.findByDisplayValue('myws')).toBeTruthy()
    expect(screen.getByPlaceholderText('https://… (self-hosted)')).toBeTruthy()
    expect(screen.getByText('Production')).toBeTruthy()
    expect(screen.getByText('Self-hosted Honcho URL.')).toBeTruthy()
  })

  it('hides fields that are not marked inline', async () => {
    await renderPanel()

    await screen.findByDisplayValue('myws')
    expect(screen.queryByDisplayValue('async')).toBeNull()
    expect(screen.queryByText('Write frequency')).toBeNull()
  })

  it('collapses and expands the fields', async () => {
    await renderPanel()

    expect(await screen.findByDisplayValue('myws')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: /Honcho settings/ }))
    expect(screen.queryByDisplayValue('myws')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: /Honcho settings/ }))
    expect(await screen.findByDisplayValue('myws')).toBeTruthy()
  })

  it('autosaves a text field on blur as a one-key partial save', async () => {
    await renderPanel()

    const baseUrl = await screen.findByPlaceholderText('https://… (self-hosted)')
    fireEvent.change(baseUrl, { target: { value: 'http://localhost:8000' } })
    fireEvent.blur(baseUrl)

    await waitFor(() =>
      expect(saveMemoryProviderConfig).toHaveBeenCalledWith('honcho', { baseUrl: 'http://localhost:8000' })
    )
    expect(saveMemoryProviderConfig).toHaveBeenCalledTimes(1)
  })

  it('does not save on blur when nothing changed', async () => {
    await renderPanel()

    const workspace = await screen.findByDisplayValue('myws')
    fireEvent.blur(workspace)

    await waitFor(() => expect(screen.queryByRole('button', { name: 'Save' })).toBeNull())
    expect(saveMemoryProviderConfig).not.toHaveBeenCalled()
  })

  it('autosaves a committed secret and clears the draft', async () => {
    await renderPanel()

    const apiKey = await screen.findByPlaceholderText('Enter Honcho API key')
    fireEvent.blur(apiKey)
    expect(saveMemoryProviderConfig).not.toHaveBeenCalled()

    fireEvent.change(apiKey, { target: { value: 'hch-new-key' } })
    fireEvent.blur(apiKey)

    await waitFor(() => expect(saveMemoryProviderConfig).toHaveBeenCalledWith('honcho', { apiKey: 'hch-new-key' }))
    await waitFor(() => expect((apiKey as HTMLInputElement).value).toBe(''))
  })

  it('offers a full-config trigger when modal-only fields exist', async () => {
    await renderPanel()

    await screen.findByDisplayValue('myws')
    expect(screen.getByRole('button', { name: /Full config/ })).toBeTruthy()
  })

  it('shows an inline error with retry when the load fails, then recovers', async () => {
    getMemoryProviderConfig.mockRejectedValueOnce(new Error('Timed out connecting to Hermes backend'))

    await renderPanel()

    expect(await screen.findByText(/Timed out connecting/)).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))

    expect(await screen.findByDisplayValue('myws')).toBeTruthy()
  })

  it('renders nothing for a provider with no declared config surface', async () => {
    getMemoryProviderConfig.mockResolvedValue({ name: 'builtin', label: 'builtin', docs_url: '', fields: [] })

    const { container } = await renderPanel('builtin')

    await waitFor(() => expect(getMemoryProviderConfig).toHaveBeenCalledWith('builtin'))
    expect(container.querySelector('section')).toBeNull()
  })
})

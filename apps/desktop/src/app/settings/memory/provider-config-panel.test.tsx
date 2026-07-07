import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { MemoryProviderConfig } from '@/types/hermes'

const getMemoryProviderConfig = vi.fn()
const saveMemoryProviderConfig = vi.fn()
const runMemoryProviderAction = vi.fn()

vi.mock('@/hermes', () => ({
  getMemoryProviderConfig: (provider: string) => getMemoryProviderConfig(provider),
  saveMemoryProviderConfig: (provider: string, values: unknown) => saveMemoryProviderConfig(provider, values),
  runMemoryProviderAction: (provider: string, action: string, values: unknown) =>
    runMemoryProviderAction(provider, action, values)
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
    actions: [],
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
  runMemoryProviderAction.mockResolvedValue({ ok: true, result: { message: 'pong' } })
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

  it('saves only inline values, with a blank secret', async () => {
    await renderPanel()

    const baseUrl = await screen.findByPlaceholderText('https://… (self-hosted)')
    fireEvent.change(baseUrl, { target: { value: 'http://localhost:8000' } })
    fireEvent.change(screen.getByDisplayValue('myws'), { target: { value: 'ben-bank' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    await waitFor(() =>
      expect(saveMemoryProviderConfig).toHaveBeenCalledWith('honcho', {
        apiKey: '',
        baseUrl: 'http://localhost:8000',
        environment: 'production',
        workspace: 'ben-bank'
      })
    )
  })

  it('offers a full-config trigger when modal-only fields exist', async () => {
    await renderPanel()

    await screen.findByDisplayValue('myws')
    expect(screen.getByRole('button', { name: /Full config/ })).toBeTruthy()
  })

  it('renders declared actions as buttons and posts the inline values', async () => {
    const schema = honchoSchema()
    schema.actions = [{ key: 'ping', label: 'Ping', description: 'Check the server' }]
    getMemoryProviderConfig.mockResolvedValue(schema)

    await renderPanel()

    fireEvent.click(await screen.findByRole('button', { name: 'Ping' }))

    await waitFor(() =>
      expect(runMemoryProviderAction).toHaveBeenCalledWith('honcho', 'ping', expect.objectContaining({ workspace: 'myws' }))
    )
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

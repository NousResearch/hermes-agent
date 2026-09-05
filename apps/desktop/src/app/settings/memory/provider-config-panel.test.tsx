import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { MemoryProviderConfig } from '@/types/hermes'

const getMemoryProviderConfig = vi.fn()
const runMemoryProviderAction = vi.fn()
const saveMemoryProviderConfig = vi.fn()

vi.mock('@/hermes', () => ({
  getMemoryProviderConfig: (provider: string, profile?: null | string) => getMemoryProviderConfig(provider, profile),
  runMemoryProviderAction: (provider: string, action: string, payload: unknown, profile?: null | string) =>
    runMemoryProviderAction(provider, action, payload, profile),
  saveMemoryProviderConfig: (provider: string, values: unknown, profile?: null | string) =>
    saveMemoryProviderConfig(provider, values, profile)
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

function managedSchema(): MemoryProviderConfig {
  return {
    description: 'Configure Example Memory.',
    docs_url: '',
    fields: [
      {
        description: '',
        group: '',
        inline: false,
        is_set: true,
        key: 'name',
        kind: 'text',
        label: 'Profile name',
        options: [],
        placeholder: '',
        required: true,
        value: 'primary'
      }
    ],
    label: 'Example Memory',
    name: 'example',
    status_action: 'health',
    submit_action: 'save',
    summary: {
      items: [{ label: 'Active profile', value: 'primary' }],
      status: { label: 'Checking', message: '', state: 'checking' }
    }
  }
}

beforeEach(() => {
  getMemoryProviderConfig.mockResolvedValue(honchoSchema())
  runMemoryProviderAction.mockResolvedValue({ ok: true })
  saveMemoryProviderConfig.mockResolvedValue({ ok: true })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

async function renderPanel(provider = 'honcho', profile?: string) {
  const { ProviderConfigPanel } = await import('./provider-config-panel')

  return render(<ProviderConfigPanel profile={profile} provider={provider} />)
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
      expect(saveMemoryProviderConfig).toHaveBeenCalledWith('honcho', { baseUrl: 'http://localhost:8000' }, undefined)
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

    await waitFor(() =>
      expect(saveMemoryProviderConfig).toHaveBeenCalledWith('honcho', { apiKey: 'hch-new-key' }, undefined)
    )
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

  it('ignores a stale response after the selected provider changes', async () => {
    let resolveHoncho: ((value: MemoryProviderConfig) => void) | undefined

    const delayedHoncho = new Promise<MemoryProviderConfig>(resolve => {
      resolveHoncho = resolve
    })

    const example = managedSchema()

    getMemoryProviderConfig.mockImplementation((provider: string) =>
      provider === 'honcho' ? delayedHoncho : Promise.resolve(example)
    )

    const { rerender } = await renderPanel('honcho')
    const { ProviderConfigPanel } = await import('./provider-config-panel')

    rerender(<ProviderConfigPanel provider="example" />)

    expect(await screen.findByRole('button', { name: 'Configure' })).toBeTruthy()

    resolveHoncho?.(honchoSchema())

    await waitFor(() => expect(screen.queryByText('Honcho settings')).toBeNull())
    expect(screen.getByText('Example Memory settings')).toBeTruthy()
  })

  it('renders nothing for a provider with no declared config surface', async () => {
    getMemoryProviderConfig.mockResolvedValue({ name: 'builtin', label: 'builtin', docs_url: '', fields: [] })

    const { container } = await renderPanel('builtin')

    await waitFor(() => expect(getMemoryProviderConfig).toHaveBeenCalledWith('builtin', undefined))
    expect(container.querySelector('section')).toBeNull()
  })

  it('renders managed settings before health completes and rechecks health after save', async () => {
    let resolveFirstHealth: ((value: unknown) => void) | undefined

    getMemoryProviderConfig.mockResolvedValue(managedSchema())
    runMemoryProviderAction.mockImplementation((_provider: string, action: string) => {
      if (action === 'health' && !resolveFirstHealth) {
        return new Promise(resolve => (resolveFirstHealth = resolve))
      }

      if (action === 'health') {
        return Promise.resolve({ label: 'Healthy', message: 'Ready', state: 'healthy' })
      }

      return Promise.resolve({ ok: true })
    })

    await renderPanel('example')

    expect(await screen.findByRole('button', { name: 'Configure' })).toBeTruthy()
    const summary = screen.getByRole('group', { name: 'Example Memory connection summary' })

    expect(summary.parentElement?.className).toContain('@container')
    expect(summary.className).toContain('@2xl:grid-cols')
    expect(screen.getByText('Checking')).toBeTruthy()

    resolveFirstHealth?.({ label: 'Healthy', message: 'Ready', state: 'healthy' })
    expect(await screen.findByText('Healthy')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Configure' }))
    fireEvent.click(await screen.findByRole('button', { name: 'Save setup' }))

    await waitFor(() => {
      expect(runMemoryProviderAction.mock.calls.filter(([, action]) => action === 'health')).toHaveLength(2)
    })
  })

  it('scopes managed config and status requests to the selected settings profile', async () => {
    getMemoryProviderConfig.mockResolvedValue(managedSchema())
    runMemoryProviderAction.mockResolvedValue({ label: 'Healthy', message: 'Ready', state: 'healthy' })

    await renderPanel('example', 'work')

    await waitFor(() => expect(getMemoryProviderConfig).toHaveBeenCalledWith('example', 'work'))
    await waitFor(() => expect(runMemoryProviderAction).toHaveBeenCalledWith('example', 'health', {}, 'work'))
  })

  it.each(['Save setup', 'Start server'])('keeps the managed form open when refresh fails after %s', async button => {
    const schema = managedSchema()
    schema.actions = [
      {
        name: 'start',
        label: 'Start server',
        description: '',
        after_field: 'name',
        payload_fields: ['name'],
        refresh_after: true,
        visible_when: []
      }
    ]
    getMemoryProviderConfig.mockResolvedValueOnce(schema).mockRejectedValueOnce(new Error('Could not reload settings'))
    runMemoryProviderAction.mockImplementation((_provider: string, action: string) =>
      Promise.resolve(action === 'health' ? { label: 'Healthy', message: '', state: 'healthy' } : { ok: true })
    )

    await renderPanel('example')
    fireEvent.click(await screen.findByRole('button', { name: 'Configure' }))
    fireEvent.change(await screen.findByDisplayValue('primary'), { target: { value: 'edited-profile' } })
    fireEvent.click(screen.getByRole('button', { name: button }))

    expect((await screen.findByRole('alert')).textContent).toBe('Could not reload settings')
    expect(screen.getByRole('dialog', { name: 'Configure Example Memory' })).toBeTruthy()
    expect(screen.getByDisplayValue('edited-profile')).toBeTruthy()
    expect(getMemoryProviderConfig).toHaveBeenCalledTimes(2)
  })
})

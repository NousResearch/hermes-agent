import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type { MemoryProviderConfig, MemoryProviderField } from '@/types/hermes'

const getMemoryProviderConfig = vi.fn()
const notify = vi.fn()
const runMemoryProviderAction = vi.fn()

class TestResizeObserver {
  disconnect() {}
  observe() {}
  unobserve() {}
}

beforeAll(() => {
  vi.stubGlobal('ResizeObserver', TestResizeObserver)
  Element.prototype.scrollIntoView = vi.fn()
})

vi.mock('@/hermes', () => ({
  getMemoryProviderConfig: (provider: string, profile?: null | string) => getMemoryProviderConfig(provider, profile),
  runMemoryProviderAction: (provider: string, action: string, payload: unknown, profile?: null | string) =>
    runMemoryProviderAction(provider, action, payload, profile)
}))

vi.mock('@/store/notifications', () => ({
  notify: (notification: unknown) => notify(notification)
}))

function field(
  overrides: Partial<MemoryProviderField> & Pick<MemoryProviderField, 'key' | 'kind'>
): MemoryProviderField {
  return {
    description: '',
    group: '',
    inline: false,
    is_set: false,
    label: overrides.key,
    options: [],
    placeholder: '',
    value: '',
    ...overrides
  }
}

function schema(): MemoryProviderConfig {
  const service = [{ key: 'setup_type', pattern: '', values: ['service'] }]
  const profile = [{ key: 'setup_type', pattern: '', values: ['profile'] }]
  const custom = [{ key: 'setup_type', pattern: '', values: ['custom'] }]

  return {
    actions: [
      {
        after_field: 'url',
        description: '',
        label: 'Start local server',
        name: 'start-local',
        payload_fields: ['url'],
        refresh_after: true,
        visible_when: [...custom, { key: 'url', pattern: '^http://127\\.0\\.0\\.1', values: [] }]
      }
    ],
    description: 'Configure the provider.',
    docs_url: '',
    fields: [
      field({
        key: 'setup_type',
        kind: 'segmented',
        label: 'Setup type',
        options: [
          { description: '', label: 'Managed Service', value: 'service' },
          { description: '', label: 'Existing Profiles', value: 'profile' },
          { description: '', label: 'Custom Server', value: 'custom' }
        ],
        required: true,
        value: 'service'
      }),
      field({
        dynamic_options: true,
        key: 'profile_path',
        kind: 'select',
        label: 'Provider profile',
        options: [
          {
            description: 'https://memory.example (/tmp/provider.conf)',
            label: 'primary',
            value: '/tmp/provider.conf'
          }
        ],
        required: true,
        search_placeholder: 'Search profiles...',
        searchable: true,
        value: '/tmp/provider.conf',
        visible_when: profile
      }),
      field({
        key: 'api_key',
        kind: 'secret',
        label: 'API key',
        required: true,
        visible_when: service
      }),
      field({
        key: 'url',
        kind: 'text',
        label: 'Server URL',
        required: true,
        value: 'http://127.0.0.1:1933',
        visible_when: custom
      })
    ],
    label: 'Example Memory',
    name: 'example',
    submit_action: 'save',
    submit_label: 'Save setup'
  }
}

async function renderModal(config = schema(), profile: null | string = null) {
  const { ProviderManagedConfigModal } = await import('./provider-managed-config-modal')
  const onOpenChange = vi.fn()
  const onSaved = vi.fn().mockResolvedValue(undefined)

  render(
    <ProviderManagedConfigModal
      config={config}
      onOpenChange={onOpenChange}
      onSaved={onSaved}
      open
      profile={profile}
      provider="example"
    />
  )

  return { onOpenChange, onSaved }
}

beforeEach(() => {
  getMemoryProviderConfig.mockResolvedValue(schema())
  runMemoryProviderAction.mockResolvedValue({ ok: true })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('ProviderManagedConfigModal', () => {
  it('shows only the fields for the selected setup type', async () => {
    await renderModal()

    expect(await screen.findByText('API key')).toBeTruthy()
    expect(screen.queryByText('Provider profile')).toBeNull()
    expect(screen.queryByText('Server URL')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Custom Server' }))

    expect(await screen.findByText('Server URL')).toBeTruthy()
    expect(screen.queryByText('API key')).toBeNull()
  })

  it('validates required visible fields before calling the provider action', async () => {
    await renderModal()

    fireEvent.click(await screen.findByRole('button', { name: 'Save setup' }))

    expect((await screen.findByRole('alert')).textContent).toContain('Complete the required fields before saving.')
    expect(runMemoryProviderAction).not.toHaveBeenCalled()
  })

  it('shows which dynamic profile configuration is selected', async () => {
    await renderModal()

    fireEvent.click(screen.getByRole('button', { name: 'Existing Profiles' }))

    expect(await screen.findByText('https://memory.example (/tmp/provider.conf)')).toBeTruthy()
    expect(document.querySelector('[data-slot="searchable-select-trigger"]')).not.toBeNull()
    fireEvent.click(screen.getByRole('combobox'))
    expect(screen.getByPlaceholderText('Search profiles...')).toBeTruthy()
    expect(document.querySelector('[data-slot="dialog-content"]')?.className).toContain('overflow-visible')
    expect(document.querySelector('[data-slot="provider-managed-config-scroll"]')?.className).toContain(
      'overflow-y-auto'
    )
  })

  it('submits the complete form once and explains when the setup takes effect', async () => {
    let resolveSave: ((value: { ok: boolean }) => void) | undefined

    runMemoryProviderAction.mockImplementationOnce(
      () => new Promise<{ ok: boolean }>(resolve => (resolveSave = resolve))
    )
    const { onOpenChange, onSaved } = await renderModal()
    const apiKey = await screen.findByLabelText('API key')

    fireEvent.change(apiKey, { target: { value: 'secret' } })
    const save = screen.getByRole('button', { name: 'Save setup' })
    fireEvent.click(save)
    fireEvent.click(save)

    expect(runMemoryProviderAction).toHaveBeenCalledTimes(1)
    expect(runMemoryProviderAction).toHaveBeenCalledWith(
      'example',
      'save',
      {
        overwrite: false,
        values: expect.objectContaining({ api_key: 'secret', setup_type: 'service' })
      },
      null
    )

    resolveSave?.({ ok: true })

    await waitFor(() => expect(onSaved).toHaveBeenCalledTimes(1))
    expect(onOpenChange).toHaveBeenCalledWith(false)
    expect(notify).toHaveBeenCalledWith(
      expect.objectContaining({
        message: 'This setup is active now. New messages in existing and new chats will use it.'
      })
    )
  })

  it('requires explicit confirmation before overwriting a different saved profile', async () => {
    runMemoryProviderAction.mockRejectedValueOnce(new Error('409: {"detail":"Profile exists"}'))
    const { onSaved } = await renderModal()

    fireEvent.change(await screen.findByLabelText('API key'), { target: { value: 'secret' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save setup' }))

    expect(await screen.findByText('Example Memory profile already exists')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Replace profile' }))

    await waitFor(() => expect(runMemoryProviderAction).toHaveBeenCalledTimes(2))
    expect(runMemoryProviderAction).toHaveBeenLastCalledWith(
      'example',
      'save',
      expect.objectContaining({ overwrite: true }),
      null
    )
    await waitFor(() => expect(onSaved).toHaveBeenCalledTimes(1))
  })

  it('dispatches conditional provider actions and refreshes status after success', async () => {
    const { onSaved } = await renderModal()

    fireEvent.click(await screen.findByRole('button', { name: 'Custom Server' }))
    fireEvent.click(await screen.findByRole('button', { name: 'Start local server' }))

    await waitFor(() =>
      expect(runMemoryProviderAction).toHaveBeenCalledWith(
        'example',
        'start-local',
        {
          url: 'http://127.0.0.1:1933'
        },
        null
      )
    )
    await waitFor(() => expect(onSaved).toHaveBeenCalledTimes(1))
  })

  it('refreshes dynamic options without resetting other drafts', async () => {
    const refreshed = schema()
    const profileField = refreshed.fields.find(candidate => candidate.key === 'profile_path')

    if (!profileField) {
      throw new Error('Test schema is missing profile_path')
    }

    profileField.options = [{ description: '/tmp/new.conf', label: 'new', value: '/tmp/new.conf' }]
    profileField.value = '/tmp/new.conf'
    getMemoryProviderConfig.mockResolvedValueOnce(refreshed)
    await renderModal()

    fireEvent.click(await screen.findByRole('button', { name: 'Existing Profiles' }))
    fireEvent.click(await screen.findByRole('button', { name: 'Refresh' }))

    await waitFor(() => expect(getMemoryProviderConfig).toHaveBeenCalledWith('example', null))
    expect(await screen.findByText('new')).toBeTruthy()
  })

  it('scopes dynamic refresh and submit actions to the selected settings profile', async () => {
    await renderModal(schema(), 'work')

    fireEvent.change(await screen.findByLabelText('API key'), { target: { value: 'secret' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save setup' }))

    await waitFor(() =>
      expect(runMemoryProviderAction).toHaveBeenCalledWith(
        'example',
        'save',
        expect.objectContaining({ overwrite: false }),
        'work'
      )
    )
  })
})

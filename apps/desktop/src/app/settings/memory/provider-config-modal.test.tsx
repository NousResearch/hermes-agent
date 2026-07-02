import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { MemoryProviderConfig, MemoryProviderField } from '@/types/hermes'

const saveMemoryProviderConfig = vi.fn()

vi.mock('@/hermes', () => ({
  saveMemoryProviderConfig: (provider: string, values: unknown) => saveMemoryProviderConfig(provider, values)
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

function field(overrides: Partial<MemoryProviderField> & Pick<MemoryProviderField, 'key' | 'kind'>): MemoryProviderField {
  return {
    label: overrides.key,
    value: '',
    description: '',
    placeholder: '',
    is_set: false,
    inline: false,
    group: 'Other',
    options: [],
    ...overrides
  }
}

function schema(): MemoryProviderConfig {
  return {
    name: 'honcho',
    label: 'Honcho',
    docs_url: 'https://docs.honcho.dev/v3/guides/integrations/hermes',
    fields: [
      field({ key: 'workspace', kind: 'text', label: 'Workspace', value: 'myws', inline: true, group: 'Connection' }),
      field({ key: 'saveMessages', kind: 'bool', label: 'Save messages', value: 'true', group: 'Message writing' }),
      field({ key: 'dialecticMaxChars', kind: 'number', label: 'Max result chars', value: '1200', group: 'Dialectic' }),
      field({ key: 'userPeerAliases', kind: 'json', label: 'User peer aliases', value: '{"t":"eri"}', group: 'Identity' })
    ]
  }
}

beforeEach(() => {
  saveMemoryProviderConfig.mockResolvedValue({ ok: true })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

async function renderModal(open = true) {
  const { ProviderConfigModal } = await import('./provider-config-modal')
  const onOpenChange = vi.fn()
  const onSaved = vi.fn().mockResolvedValue(undefined)
  const result = render(
    <ProviderConfigModal
      config={schema()}
      onOpenChange={onOpenChange}
      onSaved={onSaved}
      open={open}
      provider="honcho"
    />
  )
  return { ...result, onOpenChange, onSaved }
}

describe('ProviderConfigModal', () => {
  it('renders every field grouped, including inline ones, with kind-specific controls', async () => {
    await renderModal()

    expect(await screen.findByText('Message writing')).toBeTruthy()
    expect(screen.getByText('Dialectic')).toBeTruthy()
    // bool -> switch, number -> spinbutton, json/text -> textbox
    expect(screen.getByRole('switch')).toBeTruthy()
    expect(screen.getByDisplayValue('1200')).toBeTruthy()
    expect(screen.getByDisplayValue('myws')).toBeTruthy()
    expect(screen.getByDisplayValue('{"t":"eri"}')).toBeTruthy()
  })

  it('saves all fields, serializing the toggled bool to "false"', async () => {
    const { onSaved, onOpenChange } = await renderModal()

    fireEvent.click(await screen.findByRole('switch'))
    fireEvent.click(screen.getByRole('button', { name: 'Save all' }))

    await waitFor(() =>
      expect(saveMemoryProviderConfig).toHaveBeenCalledWith('honcho', {
        workspace: 'myws',
        saveMessages: 'false',
        dialecticMaxChars: '1200',
        userPeerAliases: '{"t":"eri"}'
      })
    )
    await waitFor(() => expect(onSaved).toHaveBeenCalled())
    expect(onOpenChange).toHaveBeenCalledWith(false)
  })

  it('renders nothing while closed', async () => {
    await renderModal(false)
    expect(screen.queryByText('Message writing')).toBeNull()
  })
})

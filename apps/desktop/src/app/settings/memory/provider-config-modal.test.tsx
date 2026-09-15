import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection } from '@/api/client'
import { getMemoryProviderConfig, saveMemoryProviderConfig as saveConfig } from '@/api/system'
import type { HermesApiRequest } from '@/global'
import { notify, notifyError } from '@/store/notifications'
import { $connection } from '@/store/session'
import type { MemoryProviderConfig, MemoryProviderField } from '@/types/hermes'

const saveMemoryProviderConfig = vi.fn()

vi.mock('@/hermes', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  saveMemoryProviderConfig: (provider: string, values: unknown) => saveMemoryProviderConfig(provider, values)
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

function field(
  overrides: Partial<MemoryProviderField> & Pick<MemoryProviderField, 'key' | 'kind'>
): MemoryProviderField {
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
      field({
        key: 'userPeerAliases',
        kind: 'json',
        label: 'User peer aliases',
        value: '{"t":"eri"}',
        group: 'Identity'
      })
    ]
  }
}

beforeEach(() => {
  saveMemoryProviderConfig.mockResolvedValue({ ok: true })
})

afterEach(() => {
  cleanup()
  $connection.set(null)
  setApiRequestConnection(null)
  vi.unstubAllGlobals()
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

  it('saves only edited fields, serializing the toggled bool to "false"', async () => {
    const { onSaved, onOpenChange } = await renderModal()

    fireEvent.click(await screen.findByRole('switch'))
    fireEvent.click(screen.getByRole('button', { name: 'Save changes' }))

    // A save must never ratify rendered defaults the backend does not store.
    await waitFor(() => expect(saveMemoryProviderConfig).toHaveBeenCalledWith('honcho', { saveMessages: 'false' }))
    await waitFor(() => expect(onSaved).toHaveBeenCalled())
    expect(onOpenChange).toHaveBeenCalledWith(false)
  })

  it('does not refresh or publish a delayed save after its legacy gateway owner retires', async () => {
    let finish!: (value: { ok: boolean }) => void

    const held = new Promise<{ ok: boolean }>(resolve => {
      finish = resolve
    })

    const api = vi.fn(async (request: HermesApiRequest) => {
      if (request.method === 'PUT') {
        return held
      }

      return schema()
    })

    vi.stubGlobal('hermesDesktop', { api })
    setApiRequestConnection('fixture-old')
    saveMemoryProviderConfig.mockImplementation(saveConfig)
    const { onSaved, onOpenChange } = await renderModal()
    onSaved.mockImplementation(() => getMemoryProviderConfig('honcho'))
    fireEvent.click(screen.getByRole('button', { name: 'Save changes' }))
    expect(api.mock.calls[0][0]).toMatchObject({ method: 'PUT', connectionId: 'fixture-old' })

    act(() => {
      setApiRequestConnection('fixture-new')
      $connection.set({
        baseUrl: 'https://example.invalid',
        wsUrl: 'wss://example.invalid',
        mode: 'remote',
        token: '',
        logs: [],
        isFullscreen: false,
        nativeOverlayWidth: 0,
        windowButtonPosition: null
      })
    })
    await act(async () => {
      finish({ ok: true })
      await held
    })
    expect(onSaved).not.toHaveBeenCalled()
    expect(onOpenChange).not.toHaveBeenCalled()
    expect(notify).not.toHaveBeenCalled()
    expect(notifyError).not.toHaveBeenCalled()
    expect(api).toHaveBeenCalledTimes(1)
  })

  it('renders nothing while closed', async () => {
    await renderModal(false)
    expect(screen.queryByText('Message writing')).toBeNull()
  })
})

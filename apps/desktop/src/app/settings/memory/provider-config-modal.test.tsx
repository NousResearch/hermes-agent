import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { MemoryProviderConfig } from '@/types/hermes'

import { field } from './test-fixtures'

const saveMemoryProviderConfig = vi.fn()

vi.mock('@/hermes', () => ({
  saveMemoryProviderConfig: (...args: unknown[]) => saveMemoryProviderConfig(...args)
}))

vi.mock('@/store/profile', async () => {
  const { atom } = await import('nanostores')

  return { $activeGatewayProfile: atom('default') }
})

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

// Load once at module scope so no test's 15s budget pays the heavy transform
// + import (the first-test timeout flake under CI load).
const { ProviderConfigModal } = await import('./provider-config-modal')

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
  vi.clearAllMocks()
})

function renderModal(open = true) {
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
  it.each([
    { supports_save_only: undefined, isActive: false },
    { supports_save_only: false, isActive: true }
  ])('refuses unsupported saves: %j', async ({ supports_save_only, isActive }) => {
    render(
      <ProviderConfigModal
        config={{ ...schema(), supports_save_only }}
        isActive={isActive}
        onOpenChange={vi.fn()}
        onSaved={vi.fn()}
        open
        provider="honcho"
      />
    )
    expect(await screen.findByRole('status')).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Save changes' })).toBeNull()
    expect(saveMemoryProviderConfig).not.toHaveBeenCalled()
  })

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
    await waitFor(() =>
      expect(saveMemoryProviderConfig).toHaveBeenCalledWith('honcho', { saveMessages: 'false' }, null, undefined, true)
    )
    await waitFor(() => expect(onSaved).toHaveBeenCalled())
    expect(onOpenChange).toHaveBeenCalledWith(false)
  })
})

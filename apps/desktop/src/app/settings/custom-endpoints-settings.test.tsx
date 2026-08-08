import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type {
  CustomEndpoint,
  CustomEndpointApiMode,
  CustomEndpointsResponse,
  CustomEndpointUpdate
} from '@/types/hermes'

const activateCustomEndpoint = vi.fn()
const deleteCustomEndpoint = vi.fn()
const getCustomEndpoints = vi.fn()
const saveCustomEndpoint = vi.fn()
const validateCustomEndpoint = vi.fn()

vi.mock('@/hermes', () => ({
  activateCustomEndpoint: (id: string) => activateCustomEndpoint(id),
  deleteCustomEndpoint: (id: string) => deleteCustomEndpoint(id),
  getCustomEndpoints: () => getCustomEndpoints(),
  saveCustomEndpoint: (endpoint: CustomEndpointUpdate) => saveCustomEndpoint(endpoint),
  validateCustomEndpoint: (endpoint: CustomEndpointUpdate) => validateCustomEndpoint(endpoint)
}))

vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

function endpoint(apiMode: CustomEndpointApiMode = ''): CustomEndpoint {
  return {
    api_mode: apiMode,
    base_url: 'https://acme.example/v1',
    discover_models: true,
    has_api_key: false,
    id: 'acme',
    model: 'acme/model',
    models: ['acme/model'],
    name: 'Acme',
    source: 'providers'
  }
}

function response(endpoints: CustomEndpoint[]): CustomEndpointsResponse {
  return {
    current: { base_url: '', model: '', provider: '' },
    endpoints
  }
}

beforeEach(() => {
  activateCustomEndpoint.mockResolvedValue({ model: 'acme/model', ok: true, provider: 'acme' })
  deleteCustomEndpoint.mockResolvedValue(response([]))
  saveCustomEndpoint.mockResolvedValue({ ...response([endpoint()]), id: 'acme', ok: true })
  validateCustomEndpoint.mockResolvedValue({ message: '', models: [], ok: true, reachable: true })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

async function renderSettings() {
  const { CustomEndpointsSettings } = await import('./custom-endpoints-settings')

  return render(<CustomEndpointsSettings />)
}

describe('CustomEndpointsSettings API mode', () => {
  it('shows all compatibility choices and defaults a new endpoint to Auto', async () => {
    getCustomEndpoints.mockResolvedValue(response([]))

    await renderSettings()

    const group = await screen.findByRole('group', { name: 'API mode' })
    expect(
      within(group)
        .getAllByRole('button')
        .map(button => button.textContent)
    ).toEqual(['Auto', 'Chat', 'Responses', 'Messages'])
    expect(within(group).getByRole('button', { name: 'Auto' }).getAttribute('aria-pressed')).toBe('true')
  })

  it('restores the saved compatibility mode when editing an endpoint', async () => {
    getCustomEndpoints.mockResolvedValue(response([endpoint('codex_responses')]))

    await renderSettings()

    expect((await screen.findByRole('button', { name: 'Responses' })).getAttribute('aria-pressed')).toBe('true')
  })

  it.each([
    ['Auto', ''] as const,
    ['Chat', 'chat_completions'] as const,
    ['Responses', 'codex_responses'] as const,
    ['Messages', 'anthropic_messages'] as const
  ])('sends %s as api_mode=%s', async (label, apiMode) => {
    const initialMode = apiMode === '' ? 'chat_completions' : ''
    getCustomEndpoints.mockResolvedValue(response([endpoint(initialMode)]))

    await renderSettings()
    fireEvent.click(await screen.findByRole('button', { name: label }))
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    await waitFor(() => expect(saveCustomEndpoint).toHaveBeenCalledWith(expect.objectContaining({ api_mode: apiMode })))
  })
})

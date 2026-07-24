import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { CustomEndpoint, CustomEndpointsResponse } from '@/types/hermes'

const activateCustomEndpoint = vi.fn()
const deleteCustomEndpoint = vi.fn()
const getCustomEndpoints = vi.fn()
const saveCustomEndpoint = vi.fn()
const validateCustomEndpoint = vi.fn()

vi.mock('@/hermes', () => ({
  activateCustomEndpoint: (id: string) => activateCustomEndpoint(id),
  deleteCustomEndpoint: (id: string, source?: string) => deleteCustomEndpoint(id, source),
  getCustomEndpoints: () => getCustomEndpoints(),
  saveCustomEndpoint: (endpoint: unknown) => saveCustomEndpoint(endpoint),
  validateCustomEndpoint: (endpoint: unknown) => validateCustomEndpoint(endpoint)
}))

vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

function endpoint(id: string, source: string, patch: Partial<CustomEndpoint> = {}): CustomEndpoint {
  return {
    base_url: `https://${id}.example/v1`,
    discover_models: true,
    has_api_key: false,
    id,
    model: `${id}/model`,
    models: [`${id}/model`],
    name: id,
    source,
    ...patch
  }
}

function response(endpoints: CustomEndpoint[]): CustomEndpointsResponse {
  return {
    current: { base_url: '', model: '', provider: '' },
    endpoints
  }
}

beforeEach(() => {
  activateCustomEndpoint.mockResolvedValue({ model: 'modern/model', ok: true, provider: 'modern' })
  deleteCustomEndpoint.mockResolvedValue(response([]))
  saveCustomEndpoint.mockResolvedValue(response([]))
  validateCustomEndpoint.mockResolvedValue({ message: '', models: [], ok: true, reachable: true })
  vi.spyOn(window, 'confirm').mockReturnValue(true)
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  vi.clearAllMocks()
})

async function renderSettings() {
  const { CustomEndpointsSettings } = await import('./custom-endpoints-settings')

  return render(<CustomEndpointsSettings />)
}

describe('CustomEndpointsSettings', () => {
  it('renders legacy custom_providers entries as delete-only summaries and refreshes after deletion', async () => {
    const legacy = endpoint('legacy-0-abcd1234', 'custom_providers', {
      api_key_preview: 'sk-...1234',
      base_url: 'https://legacy-proxy.example/v1',
      has_api_key: true,
      is_current: true,
      model: 'legacy-proxy/model',
      name: 'Legacy Proxy'
    })

    getCustomEndpoints.mockResolvedValueOnce(response([legacy])).mockResolvedValueOnce(response([]))

    await renderSettings()

    expect(await screen.findByText('Legacy config')).toBeTruthy()
    expect(screen.getByText('https://legacy-proxy.example/v1')).toBeTruthy()
    expect(screen.getByText('legacy-proxy/model')).toBeTruthy()
    expect(screen.getByText('sk-...1234')).toBeTruthy()
    expect(screen.queryByRole('button', { name: /^Legacy Proxy/ })).toBeNull()
    expect(screen.queryByRole('button', { name: 'Use' })).toBeNull()
    expect((screen.getByLabelText('Name') as HTMLInputElement).value).toBe('')

    fireEvent.click(screen.getByRole('button', { name: 'Delete Legacy Proxy' }))

    await waitFor(() => expect(deleteCustomEndpoint).toHaveBeenCalledWith('legacy-0-abcd1234', 'custom_providers'))
    await waitFor(() => expect(getCustomEndpoints).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(screen.queryByText('Legacy config')).toBeNull())
  })

  it('keeps modern and direct-config endpoints selectable with their existing actions', async () => {
    const modern = endpoint('modern', 'providers', { name: 'Modern Proxy' })
    const direct = endpoint('custom', 'direct-config', { name: 'Direct Custom' })
    getCustomEndpoints.mockResolvedValue(response([modern, direct]))

    await renderSettings()
    await screen.findByText('Modern Proxy')

    fireEvent.click(screen.getByRole('button', { name: 'New endpoint' }))
    fireEvent.click(screen.getByRole('button', { name: /^Direct Custom/ }))
    expect((screen.getByLabelText('Name') as HTMLInputElement).value).toBe('Direct Custom')

    fireEvent.click(screen.getByRole('button', { name: 'New endpoint' }))
    fireEvent.click(screen.getByRole('button', { name: /^Modern Proxy/ }))
    expect((screen.getByLabelText('Name') as HTMLInputElement).value).toBe('Modern Proxy')

    expect(screen.getAllByRole('button', { name: 'Use' })).toHaveLength(2)
    expect(screen.getByRole('button', { name: 'Delete Modern Proxy' })).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Delete Direct Custom' })).toBeNull()

    fireEvent.click(screen.getAllByRole('button', { name: 'Use' })[0])
    await waitFor(() => expect(activateCustomEndpoint).toHaveBeenCalledWith('modern'))
    await waitFor(() => expect(getCustomEndpoints).toHaveBeenCalledTimes(2))
  })
})

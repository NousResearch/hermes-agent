// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { CustomEndpointsResponse } from '@/types/hermes'

const getCustomEndpoints = vi.fn()
const saveCustomEndpoint = vi.fn()
const validateCustomEndpoint = vi.fn()
const notify = vi.fn()
const notifyError = vi.fn()
const triggerHaptic = vi.fn()

vi.mock('@/store/profile', () => ({
  $activeGatewayProfile: atom('default'),
  $profiles: atom([]),
  refreshProfiles: async () => {},
  normalizeProfileKey: (p: string | null) => p || 'default',
  profileLabel: (p: { display_name?: string; name: string }) => p.display_name || p.name
}))

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  activateCustomEndpoint: vi.fn(),
  deleteCustomEndpoint: vi.fn(),
  getCustomEndpoints: (...args: unknown[]) => getCustomEndpoints(...args),
  getProfiles: async () => ({ profiles: (await import('@/store/profile')).$profiles.get() }),
  saveCustomEndpoint: (...args: unknown[]) => saveCustomEndpoint(...args),
  setApiRequestProfile: vi.fn(),
  validateCustomEndpoint: (...args: unknown[]) => validateCustomEndpoint(...args)
}))
vi.mock('@/lib/haptics', () => ({ triggerHaptic: (...args: unknown[]) => triggerHaptic(...args) }))
vi.mock('@/store/notifications', () => ({
  notify: (...args: unknown[]) => notify(...args),
  notifyError: (...args: unknown[]) => notifyError(...args)
}))

const emptyResponse: CustomEndpointsResponse = {
  current: { base_url: '', model: '', provider: '' },
  endpoints: []
}

const savedResponse: CustomEndpointsResponse = {
  current: { base_url: 'http://profile-a.test/v1', model: 'model-a', provider: 'profile-a-endpoint' },
  endpoints: [
    {
      base_url: 'http://profile-a.test/v1',
      discover_models: true,
      has_api_key: false,
      id: 'profile-a-endpoint',
      is_current: true,
      model: 'model-a',
      models: ['model-a'],
      name: 'Profile A'
    }
  ],
  id: 'profile-a-endpoint',
  ok: true
}

beforeEach(async () => {
  const { $activeGatewayProfile, $profiles } = await import('@/store/profile')
  const { $settingsScopeOverride } = await import('@/store/settings-scope')
  $activeGatewayProfile.set('default')
  $settingsScopeOverride.set(null)
  $profiles.set([])
})

afterEach(async () => {
  cleanup()
  vi.clearAllMocks()
  const { $settingsScopeOverride } = await import('@/store/settings-scope')
  $settingsScopeOverride.set(null)
})

describe('CustomEndpointsSettings', () => {
  it('sends the chosen API mode and discovered alias metadata on Save (#93622)', async () => {
    getCustomEndpoints.mockResolvedValue(emptyResponse)
    validateCustomEndpoint.mockResolvedValue({
      message: '',
      model_details: [
        { id: 'gpt-5.6-sol' },
        { canonical_model: 'gpt-5.6-sol', id: 'gpt-5.6-sol-high', reasoning_effort: 'high' }
      ],
      models: ['gpt-5.6-sol', 'gpt-5.6-sol-high'],
      ok: true,
      reachable: true,
      transport_checked: 'codex_responses'
    })
    saveCustomEndpoint.mockResolvedValue(savedResponse)
    const { CustomEndpointsSettings } = await import('./custom-endpoints-settings')

    render(<CustomEndpointsSettings />)

    await screen.findByText('No custom endpoints')
    fireEvent.change(screen.getByPlaceholderText('Axet Proxy'), { target: { value: 'Responses gateway' } })
    fireEvent.change(screen.getByPlaceholderText('http://127.0.0.1:8081/v1'), {
      target: { value: 'https://responses-gateway.example.com/v1' }
    })
    fireEvent.click(screen.getByRole('button', { name: 'Responses API' }))
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Test' }))
    })
    fireEvent.change(screen.getByPlaceholderText('gpt-5.4'), { target: { value: 'gpt-5.6-sol-high' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    expect(validateCustomEndpoint).toHaveBeenCalledWith(
      expect.objectContaining({ api_mode: 'codex_responses' }),
      'default'
    )
    expect(notify).toHaveBeenCalledWith({
      kind: 'success',
      message: 'Endpoint is reachable (Responses API route served). Found 2 models.'
    })
    expect(saveCustomEndpoint).toHaveBeenCalledWith(
      expect.objectContaining({
        api_mode: 'codex_responses',
        model: 'gpt-5.6-sol-high',
        model_details: expect.arrayContaining([
          expect.objectContaining({ canonical_model: 'gpt-5.6-sol', id: 'gpt-5.6-sol-high', reasoning_effort: 'high' })
        ]),
        models: ['gpt-5.6-sol', 'gpt-5.6-sol-high']
      }),
      'default'
    )
  })

  it('loads and saves endpoints for the Settings Applies-to profile, not only the active bot', async () => {
    const { $activeGatewayProfile, $profiles } = await import('@/store/profile')
    const { $settingsScopeOverride } = await import('@/store/settings-scope')
    $activeGatewayProfile.set('carousel-director')
    $settingsScopeOverride.set('content-studio')
    $profiles.set(
      ['carousel-director', 'content-studio'].map(name => ({
        name,
        has_env: false,
        is_default: false,
        model: null,
        path: '',
        provider: null,
        skill_count: 0
      }))
    )
    getCustomEndpoints.mockResolvedValue(emptyResponse)
    saveCustomEndpoint.mockResolvedValue(savedResponse)
    const { CustomEndpointsSettings } = await import('./custom-endpoints-settings')

    render(<CustomEndpointsSettings />)

    await waitFor(() => expect(getCustomEndpoints).toHaveBeenCalledWith('content-studio'))
    expect(screen.getByText('Applies to')).toBeTruthy()

    fireEvent.change(screen.getByPlaceholderText('Axet Proxy'), { target: { value: 'Studio gateway' } })
    fireEvent.change(screen.getByPlaceholderText('http://127.0.0.1:8081/v1'), {
      target: { value: 'https://studio.example.com/v1' }
    })
    fireEvent.change(screen.getByPlaceholderText('gpt-5.4'), { target: { value: 'studio-model' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    expect(saveCustomEndpoint).toHaveBeenCalledWith(
      expect.objectContaining({ name: 'Studio gateway' }),
      'content-studio'
    )
  })

  it('hydrates the API mode from a saved endpoint', async () => {
    getCustomEndpoints.mockResolvedValue({
      ...savedResponse,
      endpoints: [{ ...savedResponse.endpoints[0], api_mode: 'anthropic_messages' }]
    })
    const { CustomEndpointsSettings } = await import('./custom-endpoints-settings')

    render(<CustomEndpointsSettings />)

    await screen.findByText('Profile A')
    expect(screen.getByRole('button', { name: 'Anthropic Messages' }).getAttribute('aria-pressed')).toBe('true')
  })

  it('drops a pending save completion after its profile-scoped view unmounts', async () => {
    let resolveSave!: (value: CustomEndpointsResponse) => void
    saveCustomEndpoint.mockReturnValue(new Promise(resolve => (resolveSave = resolve)))
    getCustomEndpoints.mockResolvedValue(emptyResponse)
    const onConfigSaved = vi.fn()
    const onMainModelChanged = vi.fn()
    const { CustomEndpointsSettings } = await import('./custom-endpoints-settings')

    const view = render(
      <CustomEndpointsSettings onConfigSaved={onConfigSaved} onMainModelChanged={onMainModelChanged} />
    )

    await screen.findByText('No custom endpoints')
    fireEvent.change(screen.getByPlaceholderText('Axet Proxy'), { target: { value: 'Profile A' } })
    fireEvent.change(screen.getByPlaceholderText('http://127.0.0.1:8081/v1'), {
      target: { value: 'http://profile-a.test/v1' }
    })
    fireEvent.change(screen.getByPlaceholderText('gpt-5.4'), { target: { value: 'model-a' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))
    expect(saveCustomEndpoint).toHaveBeenCalledTimes(1)

    // SettingsView keys ProvidersSettings by the selected profile, so a profile
    // switch unmounts this instance while its already-routed write is pending.
    view.unmount()
    await act(async () => resolveSave(savedResponse))

    expect(onMainModelChanged).not.toHaveBeenCalled()
    expect(onConfigSaved).not.toHaveBeenCalled()
    expect(triggerHaptic).not.toHaveBeenCalled()
    expect(notify).not.toHaveBeenCalled()
    expect(notifyError).not.toHaveBeenCalled()
  })

  it('Test rewrites the URL field to the base that actually served /models (#65488)', async () => {
    getCustomEndpoints.mockResolvedValue(emptyResponse)
    validateCustomEndpoint.mockResolvedValue({
      ok: true,
      message: '',
      models: ['model-a'],
      resolved_base_url: 'http://h.test/v1'
    })
    const { CustomEndpointsSettings } = await import('./custom-endpoints-settings')
    render(<CustomEndpointsSettings onConfigSaved={vi.fn()} onMainModelChanged={vi.fn()} />)

    await screen.findByText('No custom endpoints')
    const urlInput = screen.getByPlaceholderText<HTMLInputElement>('http://127.0.0.1:8081/v1')
    fireEvent.change(urlInput, { target: { value: 'http://h.test' } })
    await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Test' })))

    // Save stores form.baseUrl verbatim and chat POSTs {base_url}/chat/completions, so the
    // typed bare root would 404 every request even though the test looked green.
    expect(urlInput.value).toBe('http://h.test/v1')
  })
})

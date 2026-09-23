// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'
import { $settingsOwner, $settingsScopeOverride } from '@/store/settings-scope'
import type { CustomEndpoint, CustomEndpointsResponse } from '@/types/hermes'

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

import { CustomEndpointsSettings } from './custom-endpoints-settings'

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

function profile(name: string, isDefault = false) {
  return {
    has_env: false,
    is_default: isDefault,
    model: null,
    name,
    path: '',
    provider: null,
    skill_count: 0
  }
}

function currentScope() {
  const scope = $settingsOwner.get()

  expect(scope).toBeTruthy()

  return scope!
}

beforeEach(async () => {
  const { $activeGatewayProfile, $profiles } = await import('@/store/profile')
  vi.stubGlobal('hermesDesktop', {
    ...window.hermesDesktop,
    getConnectionFor: async ({ connectionId, profile: name }: { connectionId: string; profile: string }) => ({
      ...$connection.get(),
      connectionId,
      profile: name
    })
  })
  $activeGatewayProfile.set('default')
  $settingsScopeOverride.set(null)
  $profiles.set([profile('default', true)])
  $connection.set({
    authMode: 'token',
    baseUrl: 'https://gateway-a.example',
    connectionId: 'gateway',
    headers: { 'Cf-Access-Client-Id': 'client-a' },
    mode: 'remote',
    profile: 'default',
    remoteHost: 'operator@gateway-a',
    token: 'token-a'
  } as never)
})

afterEach(async () => {
  cleanup()
  vi.unstubAllGlobals()
  vi.clearAllMocks()
  const { $profiles } = await import('@/store/profile')
  $settingsScopeOverride.set(null)
  $profiles.set([])
  $connection.set(null)
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

    render(<CustomEndpointsSettings scope={currentScope()} />)

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
      expect.objectContaining({ connectionId: 'gateway', profile: 'default' })
    )
    expect(notify).toHaveBeenCalledWith(expect.objectContaining({ kind: 'success' }))
    expect(saveCustomEndpoint).toHaveBeenCalledWith(
      expect.objectContaining({
        api_mode: 'codex_responses',
        model: 'gpt-5.6-sol-high',
        model_details: expect.arrayContaining([
          expect.objectContaining({ canonical_model: 'gpt-5.6-sol', id: 'gpt-5.6-sol-high', reasoning_effort: 'high' })
        ]),
        models: ['gpt-5.6-sol', 'gpt-5.6-sol-high']
      }),
      expect.objectContaining({ connectionId: 'gateway', profile: 'default' })
    )
  })

  it('loads and saves endpoints for the Settings Applies-to profile, not only the active bot', async () => {
    const { $activeGatewayProfile, $profiles } = await import('@/store/profile')
    $activeGatewayProfile.set('carousel-director')
    $connection.set({ ...$connection.get(), profile: 'carousel-director' } as never)
    $settingsScopeOverride.set('content-studio')
    $profiles.set([profile('carousel-director'), profile('content-studio')])
    getCustomEndpoints.mockResolvedValue(emptyResponse)
    saveCustomEndpoint.mockResolvedValue(savedResponse)
    await waitFor(() => expect($settingsOwner.get()?.profile).toBe('content-studio'))

    render(<CustomEndpointsSettings scope={currentScope()} />)

    await waitFor(() =>
      expect(getCustomEndpoints).toHaveBeenCalledWith(expect.objectContaining({ profile: 'content-studio' }))
    )
    expect(screen.getByText('Applies to')).toBeTruthy()

    fireEvent.change(await screen.findByPlaceholderText('Axet Proxy'), { target: { value: 'Studio gateway' } })
    fireEvent.change(screen.getByPlaceholderText('http://127.0.0.1:8081/v1'), {
      target: { value: 'https://studio.example.com/v1' }
    })
    fireEvent.change(screen.getByPlaceholderText('gpt-5.4'), { target: { value: 'studio-model' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    expect(saveCustomEndpoint).toHaveBeenCalledWith(
      expect.objectContaining({ name: 'Studio gateway' }),
      expect.objectContaining({ profile: 'content-studio' })
    )
  })

  it('hydrates the API mode from a saved endpoint', async () => {
    getCustomEndpoints.mockResolvedValue({
      ...savedResponse,
      endpoints: [{ ...savedResponse.endpoints[0], api_mode: 'anthropic_messages' }]
    })

    render(<CustomEndpointsSettings scope={currentScope()} />)

    await screen.findByText('Profile A')
    expect(screen.getByRole('button', { name: 'Anthropic Messages' }).getAttribute('aria-pressed')).toBe('true')
  })

  it('drops a pending save completion after its profile-scoped view unmounts', async () => {
    let resolveSave!: (value: CustomEndpointsResponse) => void
    saveCustomEndpoint.mockReturnValue(new Promise(resolve => (resolveSave = resolve)))
    getCustomEndpoints.mockResolvedValue(emptyResponse)
    const onConfigSaved = vi.fn()
    const onMainModelChanged = vi.fn()

    const view = render(
      <CustomEndpointsSettings
        onConfigSaved={onConfigSaved}
        onMainModelChanged={onMainModelChanged}
        scope={currentScope()}
      />
    )

    await screen.findByText('No custom endpoints')
    fireEvent.change(screen.getByPlaceholderText('Axet Proxy'), { target: { value: 'Profile A' } })
    fireEvent.change(screen.getByPlaceholderText('http://127.0.0.1:8081/v1'), {
      target: { value: 'http://profile-a.test/v1' }
    })
    fireEvent.change(screen.getByPlaceholderText('gpt-5.4'), { target: { value: 'model-a' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))
    expect(saveCustomEndpoint).toHaveBeenCalledTimes(1)

    view.unmount()
    await act(async () => resolveSave(savedResponse))

    expect(onMainModelChanged).not.toHaveBeenCalled()
    expect(onConfigSaved).not.toHaveBeenCalled()
    expect(triggerHaptic).not.toHaveBeenCalled()
    expect(notify).not.toHaveBeenCalled()
    expect(notifyError).not.toHaveBeenCalled()
  })

  it('does not publish a late save after the originating registered owner is replaced', async () => {
    const originalEndpoint = savedResponse.endpoints[0] as CustomEndpoint
    getCustomEndpoints.mockResolvedValue({ ...savedResponse, endpoints: [originalEndpoint] })
    let resolveSave!: (value: CustomEndpointsResponse) => void
    saveCustomEndpoint.mockReturnValue(new Promise(resolve => (resolveSave = resolve)))
    const onConfigSaved = vi.fn()
    const onMainModelChanged = vi.fn()

    render(
      <CustomEndpointsSettings
        onConfigSaved={onConfigSaved}
        onMainModelChanged={onMainModelChanged}
        scope={currentScope()}
      />
    )
    await screen.findByDisplayValue('Profile A')
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))
    await waitFor(() => expect(saveCustomEndpoint).toHaveBeenCalled())

    const staleEndpoint = { ...originalEndpoint, name: 'Stale response' }
    await act(async () => {
      $connection.set({
        authMode: 'token',
        baseUrl: 'https://gateway-b.example',
        connectionId: 'gateway',
        headers: { 'Cf-Access-Client-Id': 'client-b' },
        mode: 'remote',
        profile: 'default',
        remoteHost: 'operator@gateway-b',
        token: 'token-b'
      } as never)
      resolveSave({ ...savedResponse, endpoints: [staleEndpoint], id: staleEndpoint.id })
    })

    expect(onConfigSaved).not.toHaveBeenCalled()
    expect(onMainModelChanged).not.toHaveBeenCalled()
    expect(screen.getByDisplayValue('Profile A')).toBeTruthy()
    expect(screen.queryByDisplayValue('Stale response')).toBeNull()
  })

  it('does not publish callbacks while editing a non-active profile owner', async () => {
    const { $profiles } = await import('@/store/profile')
    $profiles.set([profile('default', true), profile('beta')])
    $settingsScopeOverride.set('beta')
    getCustomEndpoints.mockResolvedValue(savedResponse)
    saveCustomEndpoint.mockResolvedValue(savedResponse)
    await waitFor(() => expect($settingsOwner.get()?.profile).toBe('beta'))
    const onConfigSaved = vi.fn()
    const onMainModelChanged = vi.fn()

    render(
      <CustomEndpointsSettings
        onConfigSaved={onConfigSaved}
        onMainModelChanged={onMainModelChanged}
        scope={currentScope()}
      />
    )
    await screen.findByDisplayValue('Profile A')
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))
    await waitFor(() => expect(saveCustomEndpoint).toHaveBeenCalled())

    expect(onConfigSaved).not.toHaveBeenCalled()
    expect(onMainModelChanged).not.toHaveBeenCalled()
  })

  it('Test rewrites the URL field to the base that actually served /models (#65488)', async () => {
    getCustomEndpoints.mockResolvedValue(emptyResponse)
    validateCustomEndpoint.mockResolvedValue({
      ok: true,
      message: '',
      models: ['model-a'],
      resolved_base_url: 'http://h.test/v1'
    })
    render(
      <CustomEndpointsSettings onConfigSaved={vi.fn()} onMainModelChanged={vi.fn()} scope={currentScope()} />
    )

    await screen.findByText('No custom endpoints')
    const urlInput = screen.getByPlaceholderText<HTMLInputElement>('http://127.0.0.1:8081/v1')
    fireEvent.change(urlInput, { target: { value: 'http://h.test' } })
    await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Test' })))

    expect(urlInput.value).toBe('http://h.test/v1')
  })
})

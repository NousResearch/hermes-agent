import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesModule from '@/hermes'
import { $activeGatewayProfile, $profiles } from '@/store/profile'
import { $connection } from '@/store/session'
import { $settingsOwner, $settingsScopeOverride } from '@/store/settings-scope'
import type { CustomEndpoint } from '@/types/hermes'

const api = vi.hoisted(() => ({
  activateCustomEndpoint: vi.fn(),
  deleteCustomEndpoint: vi.fn(),
  getCustomEndpoints: vi.fn(),
  saveCustomEndpoint: vi.fn(),
  validateCustomEndpoint: vi.fn()
}))

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesModule>()),
  ...api
}))

import { CustomEndpointsSettings } from './custom-endpoints-settings'

const endpoint: CustomEndpoint = {
  api_key_preview: null,
  base_url: 'https://models.example/v1',
  context_length: null,
  discover_models: true,
  has_api_key: false,
  id: 'fixture',
  is_current: true,
  model: 'fixture/model',
  models: ['fixture/model'],
  name: 'Fixture',
  source: 'managed'
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

beforeEach(() => {
  vi.stubGlobal('hermesDesktop', {
    ...window.hermesDesktop,
    getConnectionFor: async ({ connectionId, profile }: { connectionId: string; profile: string }) => ({
      ...$connection.get(),
      connectionId,
      profile
    })
  })
  $activeGatewayProfile.set('alpha')
  $profiles.set([profile('alpha', true), profile('beta')])
  $settingsScopeOverride.set(null)
  $connection.set({
    authMode: 'token',
    baseUrl: 'https://gateway-a.example',
    connectionId: 'gateway',
    headers: { 'Cf-Access-Client-Id': 'client-a' },
    mode: 'remote',
    remoteHost: 'operator@gateway-a',
    token: 'token-a'
  } as never)
  api.getCustomEndpoints.mockResolvedValue({ endpoints: [endpoint] })
  api.saveCustomEndpoint.mockResolvedValue({ endpoints: [endpoint], id: endpoint.id })
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  $settingsScopeOverride.set(null)
  $profiles.set([])
  $connection.set(null)
  vi.clearAllMocks()
})

describe('CustomEndpointsSettings owner isolation', () => {
  it('shows the shared Settings profile scope', async () => {
    const scope = $settingsOwner.get()

    expect(scope).toBeTruthy()
    render(<CustomEndpointsSettings scope={scope!} />)

    expect(await screen.findByText('Applies to')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'alpha' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'beta' })).toBeTruthy()
  })

  it('does not publish a late save after the originating registered owner is replaced', async () => {
    let resolveSave!: (value: { endpoints: [typeof endpoint]; id: string }) => void
    api.saveCustomEndpoint.mockReturnValue(
      new Promise(resolve => {
        resolveSave = resolve
      })
    )
    const onConfigSaved = vi.fn()
    const onMainModelChanged = vi.fn()
    const scope = $settingsOwner.get()

    expect(scope).toBeTruthy()
    render(
      <CustomEndpointsSettings onConfigSaved={onConfigSaved} onMainModelChanged={onMainModelChanged} scope={scope!} />
    )
    await screen.findByDisplayValue('Fixture')
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))
    await waitFor(() => expect(api.saveCustomEndpoint).toHaveBeenCalled())

    const staleEndpoint = { ...endpoint, name: 'Stale response' }

    await act(async () => {
      $connection.set({
        authMode: 'token',
        baseUrl: 'https://gateway-b.example',
        connectionId: 'gateway',
        headers: { 'Cf-Access-Client-Id': 'client-b' },
        mode: 'remote',
        remoteHost: 'operator@gateway-b',
        token: 'token-b'
      } as never)
      resolveSave({ endpoints: [staleEndpoint], id: staleEndpoint.id })
    })

    expect(onConfigSaved).not.toHaveBeenCalled()
    expect(onMainModelChanged).not.toHaveBeenCalled()
    expect(screen.getByDisplayValue('Fixture')).toBeTruthy()
    expect(screen.queryByDisplayValue('Stale response')).toBeNull()
  })

  it('does not publish callbacks while editing a non-active profile owner', async () => {
    $settingsScopeOverride.set('beta')
    await waitFor(() => expect($settingsOwner.get()?.profile).toBe('beta'))
    const scope = $settingsOwner.get()
    const onConfigSaved = vi.fn()
    const onMainModelChanged = vi.fn()

    expect(scope?.profile).toBe('beta')
    render(
      <CustomEndpointsSettings onConfigSaved={onConfigSaved} onMainModelChanged={onMainModelChanged} scope={scope!} />
    )
    await screen.findByDisplayValue('Fixture')
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))
    await waitFor(() => expect(api.saveCustomEndpoint).toHaveBeenCalled())

    expect(onConfigSaved).not.toHaveBeenCalled()
    expect(onMainModelChanged).not.toHaveBeenCalled()
  })

  it('drops a pending save completion after its owner-scoped view unmounts', async () => {
    let resolveSave!: (value: { endpoints: [typeof endpoint]; id: string }) => void
    api.saveCustomEndpoint.mockReturnValue(
      new Promise(resolve => {
        resolveSave = resolve
      })
    )
    const onConfigSaved = vi.fn()
    const onMainModelChanged = vi.fn()
    const scope = $settingsOwner.get()

    expect(scope).toBeTruthy()

    const view = render(
      <CustomEndpointsSettings onConfigSaved={onConfigSaved} onMainModelChanged={onMainModelChanged} scope={scope!} />
    )

    await screen.findByDisplayValue('Fixture')
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))
    await waitFor(() => expect(api.saveCustomEndpoint).toHaveBeenCalled())

    view.unmount()
    await act(async () => resolveSave({ endpoints: [endpoint], id: endpoint.id }))

    expect(onConfigSaved).not.toHaveBeenCalled()
    expect(onMainModelChanged).not.toHaveBeenCalled()
  })
})

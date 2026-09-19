// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ProfileScope } from '@/hermes'
import { $activeGatewayProfile, $profiles } from '@/store/profile'
import { $connection } from '@/store/session'
import { $settingsOwner, $settingsScopeOverride } from '@/store/settings-scope'
import type { CustomEndpointsResponse } from '@/types/hermes'

const getCustomEndpoints = vi.fn()
const saveCustomEndpoint = vi.fn()
const validateCustomEndpoint = vi.fn()
const notify = vi.fn()
const notifyError = vi.fn()
const triggerHaptic = vi.fn()
let scope: ProfileScope

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  activateCustomEndpoint: vi.fn(),
  deleteCustomEndpoint: vi.fn(),
  getCustomEndpoints: (...args: unknown[]) => getCustomEndpoints(...args),
  saveCustomEndpoint: (...args: unknown[]) => saveCustomEndpoint(...args),
  validateCustomEndpoint: (...args: unknown[]) => validateCustomEndpoint(...args)
}))
vi.mock('./profile-scope', () => ({ SettingsProfileScope: () => null }))
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

beforeEach(() => {
  $activeGatewayProfile.set('default')
  $profiles.set([
    {
      has_env: false,
      is_default: true,
      model: null,
      name: 'default',
      path: '',
      provider: null,
      skill_count: 0
    }
  ])
  $settingsScopeOverride.set(null)
  $connection.set({
    authMode: 'token',
    baseUrl: 'http://127.0.0.1:8642',
    connectionId: 'local',
    headers: {},
    mode: 'local',
    remoteHost: '',
    token: ''
  } as never)
  scope = $settingsOwner.get()!
})

afterEach(() => {
  cleanup()
  $settingsScopeOverride.set(null)
  $profiles.set([])
  $connection.set(null)
  vi.clearAllMocks()
})

describe('CustomEndpointsSettings API modes', () => {
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

    render(<CustomEndpointsSettings scope={scope} />)

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
      scope
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
      scope
    )
  })

  it('hydrates the API mode from a saved endpoint', async () => {
    getCustomEndpoints.mockResolvedValue({
      ...savedResponse,
      endpoints: [{ ...savedResponse.endpoints[0], api_mode: 'anthropic_messages' }]
    })
    const { CustomEndpointsSettings } = await import('./custom-endpoints-settings')

    render(<CustomEndpointsSettings scope={scope} />)

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
      <CustomEndpointsSettings
        onConfigSaved={onConfigSaved}
        onMainModelChanged={onMainModelChanged}
        scope={scope}
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

  it('Test rewrites the URL field to the base that actually served /models (#65488)', async () => {
    getCustomEndpoints.mockResolvedValue(emptyResponse)
    validateCustomEndpoint.mockResolvedValue({ ok: true, message: '', models: ['model-a'], resolved_base_url: 'http://h.test/v1' })
    const { CustomEndpointsSettings } = await import('./custom-endpoints-settings')
    render(
      <CustomEndpointsSettings
        onConfigSaved={vi.fn()}
        onMainModelChanged={vi.fn()}
        scope={scope}
      />
    )

    await screen.findByText('No custom endpoints')
    const urlInput = screen.getByPlaceholderText<HTMLInputElement>('http://127.0.0.1:8081/v1')
    fireEvent.change(urlInput, { target: { value: 'http://h.test' } })
    await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Test' })))

    expect(urlInput.value).toBe('http://h.test/v1')
  })
})

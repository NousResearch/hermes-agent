import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type { OpenVikingSetup } from '@/types/hermes'

const getOpenVikingSetup = vi.fn()
const saveOpenVikingSetup = vi.fn()
const validateOpenVikingSetup = vi.fn()
const startOpenVikingLocal = vi.fn()

const OPENVIKING_SERVICE_API_KEY_URL =
  'https://console.volcengine.com/vikingdb/openviking/region:openviking+cn-beijing'

vi.mock('@/hermes', () => ({
  getOpenVikingSetup: () => getOpenVikingSetup(),
  saveOpenVikingSetup: (body: unknown) => saveOpenVikingSetup(body),
  startOpenVikingLocal: (url: string) => startOpenVikingLocal(url),
  validateOpenVikingSetup: (body: unknown) => validateOpenVikingSetup(body)
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

function setupPayload(overrides: Partial<OpenVikingSetup> = {}): OpenVikingSetup {
  return {
    active: {
      account: '',
      actor_peer_id: 'hermes',
      api_key: '',
      api_key_set: false,
      source: 'hermes',
      url: 'https://api.vikingdb.cn-beijing.volces.com/openviking',
      user: ''
    },
    defaults: {
      actor_peer_id: 'hermes',
      service_url: 'https://api.vikingdb.cn-beijing.volces.com/openviking',
      url: 'http://127.0.0.1:1933'
    },
    health: {
      label: 'Healthy',
      message: 'OpenViking is reachable.',
      status: 'healthy'
    },
    legacy_env_present: [],
    local_server: {
      openviking_server_path: ''
    },
    profiles: [
      {
        account: '',
        actor_peer_id: 'profile-agent',
        api_key_set: true,
        description: 'https://vps.example (~/.openviking/ovcli.conf.VPS)',
        display_name: 'VPS',
        is_active: false,
        name: 'VPS',
        path: '/tmp/ovcli.conf.VPS',
        source: 'saved',
        url: 'https://vps.example',
        user: ''
      }
    ],
    ...overrides
  }
}

beforeEach(() => {
  getOpenVikingSetup.mockResolvedValue(setupPayload())
  saveOpenVikingSetup.mockResolvedValue({ ok: true, mode: 'profile', profile_path: '/tmp/ovcli.conf.VPS' })
  validateOpenVikingSetup.mockResolvedValue({ ok: true, message: '', role: 'user' })
  startOpenVikingLocal.mockResolvedValue({ ok: false, message: 'openviking-server was not found on PATH.' })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

async function renderPanel() {
  const { OpenVikingConfigPanel } = await import('./openviking-config-panel')

  return render(<OpenVikingConfigPanel />)
}

async function openWizard() {
  fireEvent.click(await screen.findByRole('button', { name: 'Configure' }))

  return screen.findByRole('dialog')
}

async function chooseCredential(label: string) {
  fireEvent.click(screen.getByRole('combobox', { name: 'Credential' }))
  fireEvent.click(await screen.findByRole('option', { name: label }))
}

async function chooseExistingProfiles() {
  fireEvent.click(screen.getByRole('button', { name: 'Existing Profiles' }))
  await waitFor(() =>
    expect((screen.getByRole('button', { name: 'Next' }) as HTMLButtonElement).disabled).toBe(false)
  )
  fireEvent.click(screen.getByRole('button', { name: 'Next' }))
}

describe('OpenVikingConfigPanel', () => {
  it('shows the current active OpenViking profile and opens the setup wizard', async () => {
    getOpenVikingSetup.mockResolvedValue(
      setupPayload({
        active: {
          account: '',
          actor_peer_id: 'profile-agent',
          api_key: '',
          api_key_set: true,
          ovcli_config_path: '/tmp/ovcli.conf.VPS',
          source: 'ovcli',
          url: 'https://vps.example',
          user: ''
        },
        profiles: [
          {
            account: '',
            actor_peer_id: 'profile-agent',
            api_key_set: true,
            description: 'https://vps.example (~/.openviking/ovcli.conf.VPS)',
            display_name: 'VPS',
            is_active: true,
            name: 'VPS',
            path: '/tmp/ovcli.conf.VPS',
            source: 'saved',
            url: 'https://vps.example',
            user: ''
          }
        ]
      })
    )

    await renderPanel()

    expect(await screen.findByText('Active profile')).toBeTruthy()
    expect(screen.getByText('VPS (Custom)')).toBeTruthy()
    expect(screen.getByText('https://vps.example')).toBeTruthy()
    expect(screen.getByText('Healthy')).toBeTruthy()
    expect(screen.queryByText('Ready to use')).toBeNull()
    expect(screen.queryByText(/API key configured/)).toBeNull()
    expect(screen.queryByText(/Agent ID:/)).toBeNull()

    await openWizard()

    expect(screen.getByText('Choose setup')).toBeTruthy()
    await chooseExistingProfiles()
    fireEvent.click(screen.getByRole('combobox', { name: 'OpenViking profile' }))
    expect(await screen.findByRole('option', { name: 'VPS (Active)' })).toBeTruthy()
  })

  it('labels service-backed active profiles as OpenViking Service', async () => {
    getOpenVikingSetup.mockResolvedValue(
      setupPayload({
        active: {
          account: '',
          actor_peer_id: 'hermes',
          api_key: '',
          api_key_set: true,
          ovcli_config_path: '/tmp/ovcli.conf.openstudio',
          source: 'ovcli',
          url: 'https://api.vikingdb.cn-beijing.volces.com/openviking',
          user: ''
        },
        profiles: [
          {
            account: '',
            actor_peer_id: 'hermes',
            api_key_set: true,
            description: 'https://api.vikingdb.cn-beijing.volces.com/openviking (~/.openviking/ovcli.conf.openstudio)',
            display_name: 'openstudio',
            is_active: true,
            name: 'openstudio',
            path: '/tmp/ovcli.conf.openstudio',
            source: 'saved',
            url: 'https://api.vikingdb.cn-beijing.volces.com/openviking',
            user: ''
          }
        ]
      })
    )

    await renderPanel()

    expect(await screen.findByText('openstudio (OpenViking Service)')).toBeTruthy()
  })

  it('renders unhealthy and unreachable OpenViking status as the primary summary signal', async () => {
    const detail = 'OpenViking server responded with AuthenticationError: API key rejected by the server.'
    getOpenVikingSetup.mockResolvedValue(
      setupPayload({
        health: {
          label: 'Unhealthy',
          message: detail,
          status: 'unhealthy'
        }
      })
    )

    await renderPanel()

    expect(await screen.findByText('Unhealthy')).toBeTruthy()
    expect(screen.getByText('Server reached, review details')).toBeTruthy()
    expect(screen.queryByText(detail)).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'View OpenViking status details' }))

    expect(await screen.findByText('OpenViking status details')).toBeTruthy()
    expect(screen.getByText(detail)).toBeTruthy()
  })

  it('refreshes OpenViking profiles when the setup wizard opens', async () => {
    getOpenVikingSetup
      .mockResolvedValueOnce(setupPayload())
      .mockResolvedValueOnce(setupPayload({ profiles: [] }))

    await renderPanel()
    await openWizard()

    await waitFor(() => expect(getOpenVikingSetup).toHaveBeenCalledTimes(2))
    expect(screen.getByText('No profiles found')).toBeTruthy()
  })

  it('can refresh the existing profile list from inside the wizard', async () => {
    getOpenVikingSetup
      .mockResolvedValueOnce(setupPayload())
      .mockResolvedValueOnce(setupPayload())
      .mockResolvedValueOnce(setupPayload())
      .mockResolvedValueOnce(
        setupPayload({
          profiles: [
            {
              account: '',
              actor_peer_id: 'fresh-agent',
              api_key_set: true,
              description: 'https://fresh.example (~/.openviking/ovcli.conf.fresh)',
              display_name: 'fresh',
              is_active: false,
              name: 'fresh',
              path: '/tmp/ovcli.conf.fresh',
              source: 'saved',
              url: 'https://fresh.example',
              user: ''
            }
          ]
        })
      )

    await renderPanel()
    await openWizard()

    await chooseExistingProfiles()
    fireEvent.click(screen.getByRole('button', { name: 'Refresh profiles' }))

    await waitFor(() => expect(getOpenVikingSetup).toHaveBeenCalledTimes(4))
    expect(screen.getByText('https://fresh.example (~/.openviking/ovcli.conf.fresh)')).toBeTruthy()
  })

  it('validates and links an existing profile when saving without sending secrets', async () => {
    await renderPanel()
    await openWizard()

    await chooseExistingProfiles()
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }))
    expect(screen.queryByRole('button', { name: 'Validate profile' })).toBeNull()
    expect(screen.getByText('Profile Path:')).toBeTruthy()
    expect(screen.getByText('/tmp/ovcli.conf.VPS')).toBeTruthy()
    expect(screen.queryByText('https://vps.example (~/.openviking/ovcli.conf.VPS)')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Save profile' }))

    await waitFor(() =>
      expect(validateOpenVikingSetup).toHaveBeenCalledWith({
        profile_path: '/tmp/ovcli.conf.VPS',
        values: {}
      })
    )

    await waitFor(() =>
      expect(saveOpenVikingSetup).toHaveBeenCalledWith({
        profile_path: '/tmp/ovcli.conf.VPS',
        save_mode: 'profile',
        values: {}
      })
    )
  })

  it('explains when the selected existing profile no longer exists', async () => {
    validateOpenVikingSetup.mockRejectedValue(
      new Error('400: {"detail":"OpenViking profile file was not found: /tmp/ovcli.conf.VPS"}')
    )

    await renderPanel()
    await openWizard()

    await chooseExistingProfiles()
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }))
    fireEvent.click(screen.getByRole('button', { name: 'Save profile' }))

    const alert = await screen.findByRole('alert')
    expect(alert.textContent).toContain(
      'The selected OpenViking profile no longer exists. Refresh profiles or choose another profile.'
    )
    expect(saveOpenVikingSetup).not.toHaveBeenCalled()
  })

  it('keeps the wizard open when dismissing the existing profile dropdown', async () => {
    await renderPanel()
    await openWizard()

    await chooseExistingProfiles()

    fireEvent.click(screen.getByRole('combobox', { name: 'OpenViking profile' }))
    expect(await screen.findByRole('option', { name: 'VPS' })).toBeTruthy()

    fireEvent.pointerDown(document.body)
    fireEvent.mouseDown(document.body)
    fireEvent.click(document.body)

    expect(screen.getByRole('dialog')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Continue' })).toBeTruthy()
  })

  it('refreshes stale profiles before showing Existing Profiles details', async () => {
    getOpenVikingSetup
      .mockResolvedValueOnce(setupPayload())
      .mockResolvedValueOnce(setupPayload())
      .mockResolvedValueOnce(setupPayload({
        health: {
          label: 'Profile missing',
          message: 'The linked OpenViking profile file no longer exists. Choose another profile or recreate it.',
          status: 'unreachable'
        },
        profiles: []
      }))

    await renderPanel()
    await openWizard()
    await waitFor(() => expect(getOpenVikingSetup).toHaveBeenCalledTimes(2))

    fireEvent.click(screen.getByRole('button', { name: 'Existing Profiles' }))
    await waitFor(() => expect(getOpenVikingSetup).toHaveBeenCalledTimes(3))

    expect(screen.getByText('No profiles found')).toBeTruthy()
    expect(screen.queryByRole('combobox', { name: 'OpenViking profile' })).toBeNull()
    expect((screen.getByRole('button', { name: 'Next' }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('creates an OpenViking Service profile after validation', async () => {
    await renderPanel()
    await openWizard()

    fireEvent.click(screen.getByRole('button', { name: 'Next' }))
    const apiKeyLink = screen.getByRole('link', { name: 'Get OpenViking API key' })
    expect(apiKeyLink.getAttribute('href')).toBe(OPENVIKING_SERVICE_API_KEY_URL)

    fireEvent.change(screen.getByLabelText('Agent ID'), { target: { value: 'agent' } })
    fireEvent.change(screen.getByLabelText('OpenViking API key'), { target: { value: 'service-secret' } })
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }))
    expect(screen.queryByRole('button', { name: 'Validate connection' })).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Save profile' }))

    await waitFor(() =>
      expect(validateOpenVikingSetup).toHaveBeenCalledWith({
        require_api_key: true,
        values: {
          actor_peer_id: 'agent',
          api_key: 'service-secret',
          api_key_type: 'user',
          url: 'https://api.vikingdb.cn-beijing.volces.com/openviking'
        }
      })
    )

    await waitFor(() =>
      expect(saveOpenVikingSetup).toHaveBeenCalledWith({
        profile_name: 'openviking_service',
        save_mode: 'profile',
        values: {
          actor_peer_id: 'agent',
          api_key: 'service-secret',
          api_key_type: 'user',
          url: 'https://api.vikingdb.cn-beijing.volces.com/openviking'
        }
      })
    )
  })

  it('shows validation failures as a visible alert and does not save', async () => {
    validateOpenVikingSetup.mockResolvedValueOnce({
      ok: false,
      message: 'OpenViking validation failed: [Errno 61] Connection refused',
      role: null
    })

    await renderPanel()
    await openWizard()

    fireEvent.click(screen.getByRole('button', { name: 'Custom Server' }))
    fireEvent.click(screen.getByRole('button', { name: 'Next' }))
    fireEvent.change(screen.getByLabelText('OpenViking URL'), { target: { value: 'http://127.0.0.1:1934' } })
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }))
    fireEvent.click(screen.getByRole('button', { name: 'Save profile' }))

    const alert = await screen.findByRole('alert')
    expect(alert.textContent).toContain('OpenViking could not be reached. Check that the server URL is correct and the server is running.')
    expect(saveOpenVikingSetup).not.toHaveBeenCalled()
  })

  it('uses a neutral default profile name for custom servers', async () => {
    await renderPanel()
    await openWizard()

    fireEvent.click(screen.getByRole('button', { name: 'Custom Server' }))
    fireEvent.click(screen.getByRole('button', { name: 'Next' }))
    fireEvent.change(screen.getByLabelText('OpenViking URL'), { target: { value: 'https://custom.example' } })
    await chooseCredential('User API key')
    fireEvent.change(screen.getByLabelText('OpenViking API key'), { target: { value: 'custom-secret' } })
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }))
    fireEvent.click(screen.getByRole('button', { name: 'Save profile' }))

    await waitFor(() =>
      expect(saveOpenVikingSetup).toHaveBeenCalledWith({
        profile_name: 'openviking_custom',
        save_mode: 'profile',
        values: {
          actor_peer_id: 'hermes',
          api_key: 'custom-secret',
          api_key_type: 'user',
          url: 'https://custom.example'
        }
      })
    )
  })

  it('requires account and user before validating a custom root API key', async () => {
    await renderPanel()
    await openWizard()

    fireEvent.click(screen.getByRole('button', { name: 'Custom Server' }))
    fireEvent.click(screen.getByRole('button', { name: 'Next' }))
    await chooseCredential('Root API key')
    fireEvent.change(screen.getByLabelText('OpenViking API key'), { target: { value: 'root-secret' } })

    expect((screen.getByRole('button', { name: 'Continue' }) as HTMLButtonElement).disabled).toBe(true)

    fireEvent.change(screen.getByLabelText('Account'), { target: { value: 'account-a' } })
    expect((screen.getByRole('button', { name: 'Continue' }) as HTMLButtonElement).disabled).toBe(true)

    fireEvent.change(screen.getByLabelText('User'), { target: { value: 'user-a' } })
    expect((screen.getByRole('button', { name: 'Continue' }) as HTMLButtonElement).disabled).toBe(false)
  })

  it('saves custom root API keys with root profile metadata', async () => {
    validateOpenVikingSetup.mockResolvedValueOnce({ ok: true, message: '', role: 'root' })

    await renderPanel()
    await openWizard()

    fireEvent.click(screen.getByRole('button', { name: 'Custom Server' }))
    fireEvent.click(screen.getByRole('button', { name: 'Next' }))
    fireEvent.change(screen.getByLabelText('Profile name'), { target: { value: 'VPS_ROOT' } })
    fireEvent.change(screen.getByLabelText('OpenViking URL'), { target: { value: 'https://custom.example' } })
    await chooseCredential('Root API key')
    fireEvent.change(screen.getByLabelText('OpenViking API key'), { target: { value: 'root-secret' } })
    fireEvent.change(screen.getByLabelText('Account'), { target: { value: 'account-a' } })
    fireEvent.change(screen.getByLabelText('User'), { target: { value: 'user-a' } })
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }))
    fireEvent.click(screen.getByRole('button', { name: 'Save profile' }))

    await waitFor(() =>
      expect(saveOpenVikingSetup).toHaveBeenCalledWith({
        profile_name: 'VPS_ROOT',
        save_mode: 'profile',
        values: {
          account: 'account-a',
          actor_peer_id: 'hermes',
          api_key: 'root-secret',
          api_key_type: 'root',
          root_api_key: 'root-secret',
          url: 'https://custom.example',
          user: 'user-a'
        }
      })
    )
  })

  it('prompts when a custom user API key validates as a root key', async () => {
    validateOpenVikingSetup.mockResolvedValueOnce({ ok: true, message: '', role: 'root' })

    await renderPanel()
    await openWizard()

    fireEvent.click(screen.getByRole('button', { name: 'Custom Server' }))
    fireEvent.click(screen.getByRole('button', { name: 'Next' }))
    await chooseCredential('User API key')
    fireEvent.change(screen.getByLabelText('OpenViking URL'), { target: { value: 'https://custom.example' } })
    fireEvent.change(screen.getByLabelText('OpenViking API key'), { target: { value: 'root-secret' } })
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }))
    fireEvent.click(screen.getByRole('button', { name: 'Save profile' }))

    expect(await screen.findByText('This key has root access.')).toBeTruthy()
    expect(screen.getByText(/Switch to Root API key/)).toBeTruthy()
    expect(saveOpenVikingSetup).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: 'Use as Root API key' }))

    expect(await screen.findByLabelText('Account')).toBeTruthy()
    expect(screen.getByLabelText('User')).toBeTruthy()
  })

  it('prompts when a custom root API key validates as a user key', async () => {
    validateOpenVikingSetup.mockResolvedValueOnce({ ok: true, message: '', role: 'user' })

    await renderPanel()
    await openWizard()

    fireEvent.click(screen.getByRole('button', { name: 'Custom Server' }))
    fireEvent.click(screen.getByRole('button', { name: 'Next' }))
    await chooseCredential('Root API key')
    fireEvent.change(screen.getByLabelText('OpenViking URL'), { target: { value: 'https://custom.example' } })
    fireEvent.change(screen.getByLabelText('OpenViking API key'), { target: { value: 'user-secret' } })
    fireEvent.change(screen.getByLabelText('Account'), { target: { value: 'account-a' } })
    fireEvent.change(screen.getByLabelText('User'), { target: { value: 'user-a' } })
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }))
    fireEvent.click(screen.getByRole('button', { name: 'Save profile' }))

    expect(await screen.findByText('This key is a user API key.')).toBeTruthy()
    expect(screen.getByText(/Switch to User API key/)).toBeTruthy()
    expect(saveOpenVikingSetup).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: 'Use as User API key' }))

    expect(await screen.findByRole('button', { name: 'Continue' })).toBeTruthy()
    expect(screen.queryByLabelText('Account')).toBeNull()
    expect(screen.queryByLabelText('User')).toBeNull()
  })

  it('only shows local server startup for local custom server URLs', async () => {
    await renderPanel()
    await openWizard()

    fireEvent.click(screen.getByRole('button', { name: 'Custom Server' }))
    fireEvent.click(screen.getByRole('button', { name: 'Next' }))

    expect(screen.getByRole('button', { name: 'Start local server' })).toBeTruthy()

    fireEvent.change(screen.getByLabelText('OpenViking URL'), { target: { value: 'https://custom.example' } })

    expect(screen.queryByRole('button', { name: 'Start local server' })).toBeNull()
  })

  it('shows local start guidance when openviking-server is missing', async () => {
    getOpenVikingSetup.mockResolvedValue(
      setupPayload({
        active: {
          account: '',
          actor_peer_id: 'hermes',
          api_key: '',
          api_key_set: false,
          source: 'hermes',
          url: 'http://127.0.0.1:1933',
          user: ''
        }
      })
    )

    await renderPanel()
    await openWizard()

    fireEvent.click(screen.getByRole('button', { name: 'Custom Server' }))
    fireEvent.click(screen.getByRole('button', { name: 'Next' }))
    fireEvent.click(screen.getByRole('button', { name: 'Start local server' }))

    const alert = await screen.findByRole('alert')
    expect(alert.textContent).toContain('openviking-server was not found on PATH.')
  })

  it('explains ignored legacy OpenViking environment variables', async () => {
    getOpenVikingSetup.mockResolvedValue(
      setupPayload({
        legacy_env_present: ['OPENVIKING_ENDPOINT', 'OPENVIKING_AGENT']
      })
    )

    await renderPanel()

    expect(await screen.findByText('Legacy OpenViking env vars ignored')).toBeTruthy()
    expect(screen.getByText('Save this setup to clear old variable names.')).toBeTruthy()
  })
})

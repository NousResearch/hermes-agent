import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Collect the component graph before the behavioral test deadline starts.
import { GatewaySettings } from './gateway-settings'

const { registry, activeId, selectConnection } = vi.hoisted(() => ({
  registry: { value: null as any },
  activeId: { value: 'saved-b' },
  selectConnection: vi.fn().mockResolvedValue(undefined)
}))

vi.mock('@nanostores/react', () => ({ useStore: (store: any) => store.value }))
vi.mock('@/store/connections', () => ({
  $connectionsRegistry: registry,
  $activeConnectionId: activeId,
  refreshConnectionsRegistry: vi.fn().mockResolvedValue(null),
  selectConnection,
  setConnectionsRegistry: vi.fn()
}))
vi.mock('./connections-registry', async importOriginal => ({
  ...(await importOriginal<any>()),
  ConnectionsRegistrySection: () => null
}))
const getConnectionConfig = vi.fn()
const saveConnectionConfig = vi.fn()

// This test owns the machine-level GatewaySettings contract. The managed SSH
// update section mounted below the registry has its own focused coverage
// (store/managed-updates.test.ts); keep its store subscriptions out of this
// single-purpose test.
vi.mock('./managed-updates-section', () => ({ ManagedUpdatesSection: () => null }))

const localConnection = {
  cloudOrg: '',
  envOverride: false,
  mode: 'local',
  remoteAuthMode: 'token',
  remoteOauthConnected: false,
  remoteTokenPreview: null,
  remoteTokenSet: false,
  remoteUrl: ''
}

beforeEach(() => {
  getConnectionConfig.mockResolvedValue(localConnection)
  saveConnectionConfig.mockResolvedValue(localConnection)
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { getConnectionConfig, saveConnectionConfig }
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('GatewaySettings', () => {
  it('reconnects a moved agent under its new team without replacing its saved identity or changing another default', async () => {
    const saved = {
      id: 'saved-b',
      kind: 'cloud',
      label: 'My agent',
      url: 'https://moved.example',
      authMode: 'oauth',
      org: 'old-team'
    }
    registry.value = { connections: [saved] }
    getConnectionConfig.mockResolvedValue({
      ...localConnection,
      mode: 'cloud',
      cloudOrg: 'old-team',
      remoteUrl: 'https://other.example'
    })
    const calls: string[] = []
    const oauthLogoutConnectionConfig = vi.fn(async () => {
      calls.push('logout')
    })
    const agentSignIn = vi.fn(async () => {
      calls.push('login')
      return { connected: true }
    })
    const save = vi.fn(async () => {
      calls.push('save')
    })
    const discover = vi.fn().mockResolvedValue({
      needsOrgSelection: true,
      orgs: [{ id: 'new-team', name: 'New team', role: 'OWNER' }]
    })
    Object.assign(window.hermesDesktop, {
      oauthLogoutConnectionConfig,
      connections: { save },
      cloud: { status: vi.fn().mockResolvedValue({ signedIn: true }), discover, agentSignIn }
    })
    render(<GatewaySettings embedded />)
    await screen.findByText('New team')
    discover.mockResolvedValue({
      agents: [{ id: 'moved', name: 'Moved agent', dashboardUrl: saved.url }],
      org: { id: 'new-team' }
    })
    fireEvent.click(screen.getByRole('button', { name: 'Select', exact: true }))
    await screen.findByRole('button', { name: 'Use gateway' })
    expect(save).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: 'Use gateway' }))
    await waitFor(() => expect(selectConnection).toHaveBeenCalledWith(saved.id))
    expect(calls).toEqual(['logout', 'login', 'save'])
    expect(agentSignIn).toHaveBeenCalledWith(saved.url)
    expect(save).toHaveBeenCalledWith({ ...saved, org: 'new-team' })
    expect(saveConnectionConfig).not.toHaveBeenCalled()
    registry.value = null
  })
  it('keeps saved Cloud instances usable without discovery and marks the live source, not the default', async () => {
    getConnectionConfig.mockResolvedValue({ ...localConnection, mode: 'cloud', remoteUrl: 'https://a.example' })
    registry.value = {
      connections: [
        { id: 'saved-a', kind: 'cloud', label: 'Research', url: 'https://a.example', authMode: 'oauth' },
        { id: 'saved-b', kind: 'cloud', label: 'Writing', url: 'https://b.example', authMode: 'oauth' }
      ]
    }
    const agentSignIn = vi.fn()
    const applyConnectionConfig = vi.fn()
    Object.assign(window.hermesDesktop, {
      applyConnectionConfig,
      cloud: {
        status: vi.fn().mockResolvedValue({ signedIn: false }),
        agentSignIn
      }
    })
    render(<GatewaySettings embedded />)
    const research = await screen.findByText('Research')
    const row = research.closest('[data-slot]') ?? research.parentElement!.parentElement!
    fireEvent.click(within(row as HTMLElement).getByRole('button', { name: 'Use gateway' }))
    await waitFor(() => expect(selectConnection).toHaveBeenCalledWith('saved-a'))
    expect(screen.getByText('Active in this window')).toBeTruthy()
    expect(agentSignIn).not.toHaveBeenCalled()
    expect(applyConnectionConfig).not.toHaveBeenCalled()
    registry.value = null
  })
  it('authenticates and saves only the chosen discovered instance with its friendly name', async () => {
    registry.value = null
    getConnectionConfig.mockResolvedValue({ ...localConnection, mode: 'cloud' })
    const agentSignIn = vi.fn().mockResolvedValue({ connected: true })
    const applyConnectionConfig = vi.fn().mockResolvedValue({ ...localConnection, mode: 'cloud' })
    Object.assign(window.hermesDesktop, {
      applyConnectionConfig,
      cloud: {
        status: vi.fn().mockResolvedValue({ signedIn: true }),
        agentSignIn,
        discover: vi.fn().mockResolvedValue({
          agents: [
            { id: 'new-a', name: 'Research Bot', dashboardUrl: 'https://new-a.example' },
            { id: 'new-b', name: 'Writing Bot', dashboardUrl: 'https://new-b.example' }
          ],
          org: { id: 'org-a' }
        })
      }
    })
    render(<GatewaySettings embedded />)
    const buttons = await screen.findAllByRole('button', { name: 'Connect', exact: true })
    expect(agentSignIn).not.toHaveBeenCalled()
    expect(applyConnectionConfig).not.toHaveBeenCalled()
    fireEvent.click(buttons[0])
    await waitFor(() =>
      expect(applyConnectionConfig).toHaveBeenCalledWith({
        mode: 'cloud',
        remoteAuthMode: 'oauth',
        remoteUrl: 'https://new-a.example',
        cloudOrg: 'org-a',
        cloudName: 'Research Bot'
      })
    )
    expect(agentSignIn).toHaveBeenCalledExactlyOnceWith('https://new-a.example')
    expect(applyConnectionConfig).toHaveBeenCalledTimes(1)
  })
  it('loads the machine-level connection config (no profile scoping)', async () => {
    render(<GatewaySettings />)
    expect(await screen.findByText('Local gateway')).toBeTruthy()
    expect(
      screen.getByText('Start a private Hermes backend on localhost. This is the default and works offline.')
    ).toBeTruthy()

    // The page manages the machine's gateway connections; it must load the
    // global config, never a per-profile override.
    await waitFor(() => expect(getConnectionConfig).toHaveBeenCalledWith(null))
    expect(getConnectionConfig).not.toHaveBeenCalledWith(expect.any(String))

    // The legacy per-profile scope switcher must not render.
    expect(screen.queryByText('Applies to')).toBeNull()
    expect(screen.queryByText('All profiles')).toBeNull()
    expect(screen.queryByText('Use default gateway')).toBeNull()
  })

  it('keeps session token input editable on saved remote config even when probe fails (#114856)', async () => {
    getConnectionConfig.mockResolvedValue({
      ...localConnection,
      mode: 'remote',
      remoteAuthMode: 'token',
      remoteTokenSet: true,
      remoteTokenPreview: 'test...',
      remoteUrl: 'https://gateway.example.com'
    })

    const probeConnectionConfig = vi.fn().mockRejectedValue(new Error('connection failed'))
    Object.assign(window.hermesDesktop, { probeConnectionConfig })

    render(<GatewaySettings embedded />)

    // Mode card for remote should be active
    expect(await screen.findByDisplayValue('https://gateway.example.com')).toBeTruthy()

    // Token input remains rendered and accessible so user can refresh their expired token
    const tokenInput = await screen.findByPlaceholderText(/Existing token test\.\.\./i)
    expect(tokenInput).toBeTruthy()
    expect(tokenInput.getAttribute('type')).toBe('password')

    // User can type a new token
    fireEvent.change(tokenInput, { target: { value: 'new-refreshed-token' } })
    expect((tokenInput as HTMLInputElement).value).toBe('new-refreshed-token')
  })
})

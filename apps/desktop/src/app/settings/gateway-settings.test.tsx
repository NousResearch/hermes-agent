import { GatewayReauthRequiredError } from '@rabbit/shared'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { deferred } from '@/test/deferred'

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

// Radix Select calls scrollIntoView / pointer-capture APIs jsdom lacks.
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

const getConnectionConfig = vi.fn()
const saveConnectionConfig = vi.fn()

// This test owns the machine-level GatewaySettings contract. The managed SSH
// update section mounted below the registry has its own focused coverage
// (store/managed-updates.test.ts); keep its store subscriptions out of this
// single-purpose test.
vi.mock('./managed-updates-section', () => ({ ManagedUpdatesSection: () => null }))

const localConnection = {
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
  Object.defineProperty(window, 'rabbitDesktop', {
    configurable: true,
    value: { getConnectionConfig, saveConnectionConfig }
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('GatewaySettings', () => {
  it('releases a pending save after a late probe invalidates its response', async () => {
    const saved = { ...localConnection, mode: 'remote', remoteUrl: 'https://a.example', remoteTokenSet: true }
    getConnectionConfig.mockResolvedValue(saved)
    const pendingSave = deferred<typeof saved>()
    const pendingProbe = deferred<{ reachable: boolean; authMode: string; providers: never[] }>()
    saveConnectionConfig.mockReturnValueOnce(pendingSave.promise)
    const probeConnectionConfig = vi.fn().mockReturnValue(pendingProbe.promise)

    Object.assign(window.rabbitDesktop, { probeConnectionConfig })
    render(<GatewaySettings />)
    const saveButton = (await screen.findByRole('button', { name: 'Save for next restart' })) as HTMLButtonElement
    await waitFor(() => expect(probeConnectionConfig).toHaveBeenCalledWith('https://a.example'))
    fireEvent.click(saveButton)
    expect(saveConnectionConfig).toHaveBeenCalledExactlyOnceWith({
      mode: 'remote',
      remoteUrl: 'https://a.example',
      remoteAuthMode: 'token',
      remoteToken: undefined
    })
    expect(saveButton.disabled).toBe(true)
    await act(async (): Promise<void> => pendingProbe.resolve({ reachable: true, authMode: 'oauth', providers: [] }))
    await act(async (): Promise<void> => pendingSave.resolve(saved))
    expect(saveButton.disabled).toBe(false)
    expect(screen.getByRole('button', { name: /Sign in with/ })).toBeTruthy()
    expect(screen.queryByPlaceholderText('Existing token saved')).toBeNull()
  })

  it('pre-saves OAuth before login and applies the resolved auth mode without requiring a test', async () => {
    getConnectionConfig.mockResolvedValue({ ...localConnection, mode: 'remote', remoteUrl: 'https://login.example' })
    const pendingSave = deferred<void>()
    saveConnectionConfig.mockReturnValueOnce(pendingSave.promise)
    const oauthLoginConnectionConfig = vi.fn().mockResolvedValue({ connected: true })
    const applyConnectionConfig = vi.fn().mockResolvedValue(localConnection)
    const testConnectionConfig = vi.fn()
    Object.assign(window.rabbitDesktop, {
      oauthLoginConnectionConfig,
      applyConnectionConfig,
      testConnectionConfig,
      probeConnectionConfig: vi.fn().mockResolvedValue({
        reachable: true,
        authMode: 'oauth',
        providers: [{ name: 'password', displayName: 'Username & Password', supportsPassword: true }]
      })
    })
    render(<GatewaySettings />)
    fireEvent.click(await screen.findByRole('button', { name: 'Sign in' }))
    expect(saveConnectionConfig).toHaveBeenCalledExactlyOnceWith({
      mode: 'remote',
      remoteAuthMode: 'oauth',
      remoteUrl: 'https://login.example'
    })
    expect(oauthLoginConnectionConfig).not.toHaveBeenCalled()
    await act(async (): Promise<void> => pendingSave.resolve())
    await screen.findByText('Signed in')
    expect(oauthLoginConnectionConfig).toHaveBeenCalledExactlyOnceWith('https://login.example')
    fireEvent.click(screen.getByRole('button', { name: 'Save and reconnect' }))
    await waitFor(() =>
      expect(applyConnectionConfig).toHaveBeenCalledExactlyOnceWith({
        mode: 'remote',
        remoteAuthMode: 'oauth',
        remoteUrl: 'https://login.example',
        remoteToken: undefined
      })
    )
    expect(testConnectionConfig).not.toHaveBeenCalled()
  })

  it('keeps a saved token when blank and requires consent before replacing it in plaintext', async () => {
    const saved = {
      ...localConnection,
      mode: 'remote',
      remoteUrl: 'https://a.example',
      remoteTokenSet: true,
      remoteTokenPreview: 'saved-preview',
      secureTokenStorage: false,
      remoteTokenPlainText: true
    }

    getConnectionConfig.mockResolvedValue(saved)
    saveConnectionConfig.mockResolvedValue(saved)
    const pendingSave = deferred<typeof saved>()
    saveConnectionConfig.mockReturnValueOnce(pendingSave.promise)
    Object.assign(window.rabbitDesktop, {
      probeConnectionConfig: vi.fn().mockResolvedValue({ reachable: true, authMode: 'token', providers: [] })
    })
    render(<GatewaySettings />)
    await screen.findByPlaceholderText('Existing token saved-preview')
    fireEvent.click(screen.getByRole('button', { name: 'Save for next restart' }))
    await waitFor(() =>
      expect(saveConnectionConfig).toHaveBeenCalledExactlyOnceWith({
        mode: 'remote',
        remoteUrl: 'https://a.example',
        remoteAuthMode: 'token',
        remoteToken: undefined
      })
    )
    // Flush the save's reset and probe effects before acquiring the replacement field.
    await act(async (): Promise<void> => pendingSave.resolve(saved))
    const tokenInput = await screen.findByPlaceholderText('Existing token saved-preview')
    expect(tokenInput.isConnected, 'saved credential control must survive the refresh probe').toBe(true)
    fireEvent.change(tokenInput, { target: { value: 'replacement' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save for next restart' }))
    await screen.findByText('Store the gateway token in plain text?')
    expect(saveConnectionConfig).toHaveBeenCalledTimes(1)
    fireEvent.click(screen.getByRole('button', { name: 'Save as plain text' }))
    await waitFor(() =>
      expect(saveConnectionConfig).toHaveBeenLastCalledWith({
        mode: 'remote',
        remoteUrl: 'https://a.example',
        remoteAuthMode: 'token',
        remoteToken: 'replacement',
        allowPlainTextToken: true
      })
    )
  })

  it('discards an old token test while saving the current credential-ready payload', async () => {
    getConnectionConfig.mockResolvedValue({ ...localConnection, mode: 'remote', remoteUrl: 'https://a.example' })
    const probeConnectionConfig = vi.fn().mockResolvedValue({ reachable: true, authMode: 'token', providers: [] })
    const pendingTest = deferred<{ ok: boolean; baseUrl: string }>()
    const testConnectionConfig = vi.fn().mockReturnValue(pendingTest.promise)

    Object.assign(window.rabbitDesktop, { probeConnectionConfig, testConnectionConfig })
    render(<GatewaySettings />)
    const token = await screen.findByPlaceholderText('Paste session token')
    fireEvent.change(token, { target: { value: 'old-token' } })
    fireEvent.click(screen.getByRole('button', { name: 'Test remote' }))
    expect(testConnectionConfig).toHaveBeenCalledWith({
      mode: 'remote',
      remoteUrl: 'https://a.example',
      remoteAuthMode: 'token',
      remoteToken: 'old-token'
    })
    fireEvent.change(token, { target: { value: 'new-token' } })
    await act(async (): Promise<void> => pendingTest.resolve({ ok: true, baseUrl: 'https://a.example' }))
    expect(screen.queryByText('Connected to https://a.example')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Save for next restart' }))
    await waitFor(() =>
      expect(saveConnectionConfig).toHaveBeenCalledWith(
        expect.objectContaining({
          mode: 'remote',
          remoteUrl: 'https://a.example',
          remoteAuthMode: 'token',
          remoteToken: 'new-token'
        })
      )
    )
  })
  it('loads the machine-level connection config (no profile scoping)', async () => {
    render(<GatewaySettings />)
    expect(await screen.findByText('Local gateway')).toBeTruthy()

    // The page manages the machine's gateway connections; it must load the
    // global config, never a per-profile override.
    await waitFor(() => expect(getConnectionConfig).toHaveBeenCalledWith(null))
    expect(getConnectionConfig).not.toHaveBeenCalledWith(expect.any(String))

    // The legacy per-profile scope switcher must not render.
    expect(screen.queryByText('Applies to')).toBeNull()
    expect(screen.queryByText('All profiles')).toBeNull()
    expect(screen.queryByText('Use default gateway')).toBeNull()
  })

  it('opens a focused, typeable custom SSH host input on the first "Custom" selection', async () => {
    getConnectionConfig.mockResolvedValue({
      ...localConnection,
      mode: 'ssh',
      sshHost: '',
      sshUser: '',
      sshPort: 22,
      sshKeyPath: '',
      sshRemoteRabbitPath: '',
      sshRemoteProfile: ''
    })
    const sshConfigHosts = vi.fn().mockResolvedValue({ hosts: ['github.com'] })
    Object.assign(window.rabbitDesktop, { sshConfigHosts })

    render(<GatewaySettings />)

    // With ~/.ssh/config aliases available the host field is a dropdown.
    fireEvent.click(await screen.findByRole('combobox'))
    fireEvent.click(screen.getByRole('option', { name: 'Custom (enter manually)…' }))

    // The FIRST pick swaps the dropdown for a free-text input, no round-trip
    // through another option needed.
    const hostRow = screen.getByText('Host').closest('.grid') as HTMLElement
    const input = within(hostRow).getByRole('textbox') as HTMLInputElement

    await waitFor(() => expect(document.activeElement).toBe(input))
    fireEvent.change(input, { target: { value: 'build-box' } })
    expect(input.value).toBe('build-box')

    // Clearing it and leaving the field backs out of Custom to the dropdown.
    fireEvent.change(input, { target: { value: '' } })
    fireEvent.blur(input)
    expect(await within(hostRow).findByRole('combobox')).toBeTruthy()
    expect(within(hostRow).queryByRole('textbox')).toBeNull()
  })
})

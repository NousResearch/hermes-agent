import { GatewayReauthRequiredError } from '@hermes/shared'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

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
  it('releases a pending save after a late probe invalidates its response', async () => {
    const saved = { ...localConnection, mode: 'remote', remoteUrl: 'https://a.example', remoteTokenSet: true }
    getConnectionConfig.mockResolvedValue(saved)
    const pendingSave = deferred<typeof saved>()
    const pendingProbe = deferred<{ reachable: boolean; authMode: string; providers: never[] }>()
    saveConnectionConfig.mockReturnValueOnce(pendingSave.promise)
    const probeConnectionConfig = vi.fn().mockReturnValue(pendingProbe.promise)

    Object.assign(window.hermesDesktop, { probeConnectionConfig })
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
    Object.assign(window.hermesDesktop, {
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
    Object.assign(window.hermesDesktop, {
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

    Object.assign(window.hermesDesktop, { probeConnectionConfig, testConnectionConfig })
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
  // #114856: an env-pinned remote (HERMES_DESKTOP_REMOTE_URL) whose session
  // lapsed could not be re-authenticated from Settings → Gateway at all — the
  // whole remote block (URL + probe + Authentication) was hidden behind
  // `!state.envOverride`, so the recovery card's "Gateway settings" escape led
  // to a read-only banner with no sign-in, leaving "Use local gateway" as the
  // only way back in. The env override pins the URL, not the browser session
  // (docs: "you still sign in from the Gateway settings panel"), so the
  // Authentication row must stay reachable — its controls are not env-owned.
  describe('env-override remote', () => {
    const envUrl = 'http://100.116.104.53:9191'

    const envRemote = {
      ...localConnection,
      envOverride: true,
      mode: 'remote',
      remoteAuthMode: 'token',
      remoteUrl: envUrl
    }

    const oauthProbe = {
      authMode: 'oauth',
      providers: [{ displayName: 'Nous Research', name: 'nous', supportsPassword: false }],
      reachable: true
    }

    it('reaches sign-in for a lapsed session instead of a dead env banner', async () => {
      const oauthLoginConnectionConfig = vi.fn().mockResolvedValue({ connected: true })
      const probeConnectionConfig = vi.fn().mockResolvedValue(oauthProbe)

      getConnectionConfig.mockResolvedValue({ ...envRemote, remoteOauthConnected: false })
      // Sign-in persists the URL + oauth mode before opening the login window;
      // the saved echo must stay remote or the signing sequence resets.
      saveConnectionConfig.mockResolvedValue({ ...envRemote, remoteAuthMode: 'oauth' })
      Object.assign(window.hermesDesktop, { oauthLoginConnectionConfig, probeConnectionConfig })

      render(<GatewaySettings embedded />)

      // The env override still owns the URL: the editor stays read-only.
      expect(((await screen.findByDisplayValue(envUrl)) as HTMLInputElement).disabled).toBe(true)

      fireEvent.click(await screen.findByRole('button', { name: 'Sign in with Nous Research' }))

      await waitFor(() => expect(oauthLoginConnectionConfig).toHaveBeenCalledWith(envUrl))
    })

    it('leaves a saved (non-env) remote session editable and unchanged', async () => {
      const oauthLoginConnectionConfig = vi.fn()

      getConnectionConfig.mockResolvedValue({ ...envRemote, envOverride: false, remoteOauthConnected: false })
      Object.assign(window.hermesDesktop, {
        oauthLoginConnectionConfig,
        probeConnectionConfig: vi.fn().mockResolvedValue(oauthProbe)
      })

      render(<GatewaySettings embedded />)

      expect(((await screen.findByDisplayValue(envUrl)) as HTMLInputElement).disabled).toBe(false)
      expect(await screen.findByRole('button', { name: 'Sign in with Nous Research' })).toBeTruthy()
      expect(oauthLoginConnectionConfig).not.toHaveBeenCalled()
    })
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

  // A saved cloud connection on a local-primary device has exactly one
  // recovery when its gateway session lapses: the silent portal cascade. The
  // dial's error copy sends the user to Settings → Gateways, so the "Use
  // gateway" action there must run that cascade and retry the switch — a
  // plain reauth failure toast would point at a page that cannot help.
  describe('saved cloud gateway re-auth', () => {
    const reauthError = new GatewayReauthRequiredError(
      'Reached the gateway over HTTP, but the OAuth session was rejected while minting a WebSocket ticket.'
    )

    const saved = {
      id: 'saved-a',
      kind: 'cloud',
      label: 'Research',
      url: 'https://a.example',
      authMode: 'oauth'
    }

    const mountCloudPanelWith = (cloud: Record<string, unknown>) => {
      getConnectionConfig.mockResolvedValue({ ...localConnection, mode: 'cloud', remoteUrl: saved.url })
      registry.value = { connections: [saved] }
      Object.assign(window.hermesDesktop, { cloud })
    }

    it('re-signs a lapsed saved gateway session via the portal cascade and retries the switch', async () => {
      mountCloudPanelWith({
        status: vi.fn().mockResolvedValue({ signedIn: true }),
        login: vi.fn(),
        agentSignIn: vi.fn().mockResolvedValue({ connected: true })
      })
      const oauthLogoutConnectionConfig = vi.fn().mockResolvedValue({ ok: true })
      Object.assign(window.hermesDesktop, { oauthLogoutConnectionConfig })
      selectConnection.mockRejectedValueOnce(reauthError).mockResolvedValueOnce(undefined)

      render(<GatewaySettings embedded />)
      const row = (await screen.findByText('Research')).closest('[data-slot]') as HTMLElement
      fireEvent.click(within(row).getByRole('button', { name: 'Use gateway' }))

      await waitFor(() => expect(selectConnection).toHaveBeenCalledTimes(2))
      expect(selectConnection).toHaveBeenNthCalledWith(1, saved.id)
      expect(selectConnection).toHaveBeenNthCalledWith(2, saved.id)
      expect(oauthLogoutConnectionConfig).toHaveBeenCalledWith(saved.url)
      expect(window.hermesDesktop!.cloud!.agentSignIn).toHaveBeenCalledWith(saved.url)
      // The portal session was already live: no interactive portal login.
      expect(window.hermesDesktop!.cloud!.login).not.toHaveBeenCalled()
      registry.value = null
    })

    it('fails closed instead of cascading against an empty saved dashboard URL', async () => {
      const savedWithoutUrl = {
        id: 'saved-without-url',
        kind: 'cloud',
        label: 'Incomplete',
        authMode: 'oauth'
      }

      getConnectionConfig.mockResolvedValue({ ...localConnection, mode: 'cloud', remoteUrl: '' })
      registry.value = { connections: [savedWithoutUrl] }
      const agentSignIn = vi.fn()
      const oauthLogoutConnectionConfig = vi.fn()
      Object.assign(window.hermesDesktop, {
        oauthLogoutConnectionConfig,
        cloud: {
          status: vi.fn().mockResolvedValue({ signedIn: true }),
          login: vi.fn(),
          agentSignIn
        }
      })
      selectConnection.mockRejectedValueOnce(reauthError)

      render(<GatewaySettings embedded />)
      const row = (await screen.findByText('Incomplete')).closest('[data-slot]') as HTMLElement
      fireEvent.click(within(row).getByRole('button', { name: 'Use gateway' }))

      await waitFor(() => expect(selectConnection).toHaveBeenCalledTimes(1))
      expect(oauthLogoutConnectionConfig).not.toHaveBeenCalled()
      expect(agentSignIn).not.toHaveBeenCalled()
      registry.value = null
    })

    it('does not cascade for a non-reauth switch failure', async () => {
      mountCloudPanelWith({
        status: vi.fn().mockResolvedValue({ signedIn: true }),
        login: vi.fn(),
        agentSignIn: vi.fn()
      })
      const oauthLogoutConnectionConfig = vi.fn()
      Object.assign(window.hermesDesktop, { oauthLogoutConnectionConfig })
      selectConnection.mockRejectedValueOnce(new Error('Timed out connecting to "Research".'))

      render(<GatewaySettings embedded />)
      const row = (await screen.findByText('Research')).closest('[data-slot]') as HTMLElement
      fireEvent.click(within(row).getByRole('button', { name: 'Use gateway' }))

      await waitFor(() => expect(selectConnection).toHaveBeenCalledTimes(1))
      expect(oauthLogoutConnectionConfig).not.toHaveBeenCalled()
      expect(window.hermesDesktop!.cloud!.agentSignIn).not.toHaveBeenCalled()
      registry.value = null
    })

    it('signs into the portal first when that session has lapsed too', async () => {
      mountCloudPanelWith({
        status: vi.fn().mockResolvedValue({ signedIn: false }),
        login: vi.fn().mockResolvedValue({ ok: true, signedIn: true }),
        agentSignIn: vi.fn().mockResolvedValue({ connected: true })
      })
      const oauthLogoutConnectionConfig = vi.fn().mockResolvedValue({ ok: true })
      Object.assign(window.hermesDesktop, { oauthLogoutConnectionConfig })
      selectConnection.mockRejectedValueOnce(reauthError).mockResolvedValueOnce(undefined)

      render(<GatewaySettings embedded />)
      const row = (await screen.findByText('Research')).closest('[data-slot]') as HTMLElement
      fireEvent.click(within(row).getByRole('button', { name: 'Use gateway' }))

      await waitFor(() => expect(selectConnection).toHaveBeenCalledTimes(2))
      expect(window.hermesDesktop!.cloud!.login).toHaveBeenCalledTimes(1)
      expect(window.hermesDesktop!.cloud!.agentSignIn).toHaveBeenCalledWith(saved.url)
      registry.value = null
    })
  })
})

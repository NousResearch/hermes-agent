import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { DesktopConnectionsRegistry } from '@/global'
import { _resetFleetRosterForTests, refreshFleetRoster } from '@/store/fleet-roster'
import { $connection } from '@/store/session'
import { deferred } from '@/test/deferred'

import {
  ConnectionsRegistrySection,
  findDuplicateConnection,
  normalizeGatewayUrl,
  sameBackendPeerLabel,
  sshCompositeKey
} from './connections-registry'

const list = vi.fn()
const save = vi.fn()
const remove = vi.fn()
const setLaunchMode = vi.fn()
const setPrimary = vi.fn()
const test = vi.fn()

const registry: DesktopConnectionsRegistry = {
  connections: [
    { id: 'local', kind: 'local', label: 'This device', tokenPreview: null, tokenSet: false },
    {
      authMode: 'token',
      id: 'homelab',
      kind: 'remote',
      label: 'Homelab',
      tokenPreview: '...abc123',
      tokenSet: true,
      url: 'http://homelab.lan:9119'
    }
  ],
  primary: 'local',
  secureTokenStorage: true,
  version: 2
}

beforeEach(() => {
  $connection.set({
    baseUrl: 'http://homelab.lan:9119',
    connectionId: 'homelab',
    isFullscreen: false,
    logs: [],
    mode: 'remote',
    nativeOverlayWidth: 0,
    token: 'test-token',
    windowButtonPosition: null,
    wsUrl: 'ws://homelab.lan:9119/ws'
  })
  list.mockResolvedValue(registry)
  save.mockResolvedValue({ connection: registry.connections[1], ok: true, registry })
  remove.mockResolvedValue({ ok: true, registry: { ...registry, connections: [registry.connections[0]] } })
  setLaunchMode.mockResolvedValue({ ok: true, registry: { ...registry, launchMode: 'last-used' } })
  setPrimary.mockResolvedValue({ ok: true, registry: { ...registry, primary: 'homelab' } })
  test.mockResolvedValue({ ok: true, reachable: true })
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { connections: { list, remove, save, setLaunchMode, setPrimary, test } }
  })
})

afterEach(() => {
  $connection.set(null)
  cleanup()
  vi.clearAllMocks()
})

describe('ConnectionsRegistrySection', () => {
  it('preserves, replaces and deletes stored headers through plaintext consent without selecting a source', async () => {
    const active = $connection.get()

    const withHeaders: DesktopConnectionsRegistry = {
      ...registry,
      secureTokenStorage: false,
      connections: [registry.connections[0], { ...registry.connections[1], headerNames: ['Keep', 'Replace', 'Delete'] }]
    }

    list.mockResolvedValueOnce(withHeaders)
    save.mockRejectedValueOnce(new Error('plaintext consent required'))
    const applyConnectionConfig = vi.fn()
    const select = vi.fn()
    Object.assign(window.hermesDesktop, { applyConnectionConfig })
    Object.assign(window.hermesDesktop.connections, { select })
    render(<ConnectionsRegistrySection />)
    fireEvent.click(await screen.findByRole('button', { name: 'Edit' }))
    const values = screen.getAllByPlaceholderText('Saved — leave blank to keep')
    fireEvent.change(values[1], { target: { value: 'new-header-secret' } })
    fireEvent.click(within(screen.getByDisplayValue('Delete').parentElement!).getByRole('button', { name: 'Remove' }))
    fireEvent.change(screen.getByPlaceholderText('Existing token ...abc123'), { target: { value: 'new-token' } })
    fireEvent.click(screen.getByText('Save connection'))
    await screen.findByText('Store the gateway token in plain text?')

    const expected = {
      id: 'homelab',
      kind: 'remote',
      label: 'Homelab',
      url: 'http://homelab.lan:9119',
      authMode: 'token',
      token: 'new-token',
      headers: { Keep: null, Replace: 'new-header-secret' }
    }

    expect(save).toHaveBeenCalledExactlyOnceWith(expected)
    fireEvent.click(screen.getByRole('button', { name: 'Save as plain text' }))
    await waitFor(() => expect(save).toHaveBeenLastCalledWith({ ...expected, allowPlainTextToken: true }))
    expect(applyConnectionConfig).not.toHaveBeenCalled()
    expect(select).not.toHaveBeenCalled()
    expect($connection.get()).toBe(active)
  })

  it('keeps stale browser sign-in out of a storage-only edit for a different URL', async () => {
    const active = $connection.get()
    const applyConnectionConfig = vi.fn()
    const saveConnectionConfig = vi.fn()
    const probeConnectionConfig = vi.fn().mockResolvedValue({ reachable: true, authMode: 'oauth', providers: [] })
    const pendingLogin = deferred<{ connected: boolean }>()
    const oauthLoginConnectionConfig = vi.fn().mockReturnValue(pendingLogin.promise)

    Object.assign(window.hermesDesktop, {
      applyConnectionConfig,
      saveConnectionConfig,
      probeConnectionConfig,
      oauthLoginConnectionConfig
    })
    render(<ConnectionsRegistrySection />)
    fireEvent.click(await screen.findByText('Add connection'))
    fireEvent.change(screen.getByPlaceholderText('Homelab'), { target: { value: 'New gateway' } })
    const url = screen.getByPlaceholderText('http://homelab.lan:9119')
    fireEvent.change(url, { target: { value: 'https://a.example' } })
    fireEvent.click(screen.getByRole('button', { name: /^(OAuth|Sign in)$/ }))
    fireEvent.click(await screen.findByRole('button', { name: /Sign in with/ }))
    await waitFor(() => expect(oauthLoginConnectionConfig).toHaveBeenCalledWith('https://a.example'))
    fireEvent.change(url, { target: { value: 'https://b.example' } })
    await act(async (): Promise<void> => pendingLogin.resolve({ connected: true }))
    expect(screen.queryByText('Signed in')).toBeNull()
    fireEvent.click(screen.getByText('Save connection'))
    await waitFor(() =>
      expect(save).toHaveBeenCalledWith({
        kind: 'remote',
        label: 'New gateway',
        url: 'https://b.example',
        authMode: 'oauth',
        headers: {}
      })
    )
    expect(applyConnectionConfig).not.toHaveBeenCalled()
    expect(saveConnectionConfig).not.toHaveBeenCalled()
    expect($connection.get()).toBe(active)
  })

  it('refreshes a cached roster immediately after a successful connection test', async () => {
    _resetFleetRosterForTests()
    const getAgentRoster = vi.fn().mockResolvedValue({ agents: [], sources: [] })
    Object.assign(window.hermesDesktop!, { getAgentRoster })

    try {
      await refreshFleetRoster()
      render(<ConnectionsRegistrySection />)
      await screen.findByText('Homelab')
      fireEvent.click(screen.getAllByRole('button', { name: /^test$/i })[0])
      await waitFor(() => expect(getAgentRoster).toHaveBeenCalledTimes(2))
    } finally {
      _resetFleetRosterForTests()
    }
  })
  it('distinguishes the current connection from the registry primary', async () => {
    render(<ConnectionsRegistrySection />)

    await waitFor(() => expect(screen.getByText('Homelab')).toBeTruthy())
    // Label and the managed pill share the copy, so expect both instances.
    expect(screen.getAllByText('This device').length).toBeGreaterThan(0)
    expect(screen.getByText('Current')).toBeTruthy()
    expect(screen.getAllByText('Primary').length).toBeGreaterThan(0)
    expect(list).toHaveBeenCalledTimes(1)
  })

  it('opens the add-connection editor and saves with a required label', async () => {
    render(<ConnectionsRegistrySection />)

    await waitFor(() => expect(screen.getByText('Homelab')).toBeTruthy())
    fireEvent.click(screen.getByText('Add connection'))

    // Save is disabled until a label is present.
    const saveButton = screen.getByText('Save connection').closest('button')!
    expect(saveButton.disabled).toBe(true)

    fireEvent.change(screen.getByPlaceholderText('Homelab'), { target: { value: 'Spark box' } })
    fireEvent.change(screen.getByPlaceholderText('http://homelab.lan:9119'), {
      target: { value: 'http://spark.lan:9119' }
    })
    expect(saveButton.disabled).toBe(false)
    fireEvent.click(saveButton)

    await waitFor(() => expect(save).toHaveBeenCalledTimes(1))
    expect(save.mock.calls[0][0]).toMatchObject({
      kind: 'remote',
      label: 'Spark box',
      url: 'http://spark.lan:9119'
    })
  })

  it('saves a custom remote Hermes path for SSH connections', async () => {
    render(<ConnectionsRegistrySection />)

    await waitFor(() => expect(screen.getByText('Homelab')).toBeTruthy())
    fireEvent.click(screen.getByText('Add connection'))
    fireEvent.click(screen.getByRole('button', { name: 'SSH' }))
    fireEvent.change(screen.getByPlaceholderText('Homelab'), { target: { value: 'Build host' } })
    fireEvent.change(screen.getByPlaceholderText('user@host:22'), { target: { value: 'dev@build.test:2222' } })
    fireEvent.change(screen.getByPlaceholderText('auto-detect'), {
      target: { value: '/opt/hermes/bin/hermes' }
    })
    fireEvent.click(screen.getByText('Save connection').closest('button')!)

    await waitFor(() => expect(save).toHaveBeenCalledTimes(1))
    expect(save.mock.calls[0][0]).toMatchObject({
      host: 'dev@build.test:2222',
      kind: 'ssh',
      label: 'Build host',
      remoteHermesPath: '/opt/hermes/bin/hermes'
    })
  })

  it('clears a saved remote Hermes path back to auto-detect', async () => {
    const sshRegistry: DesktopConnectionsRegistry = {
      ...registry,
      connections: [
        registry.connections[0],
        {
          host: 'build.test',
          id: 'build-host',
          kind: 'ssh',
          label: 'Build host',
          remoteHermesPath: '/opt/hermes/bin/hermes',
          tokenPreview: null,
          tokenSet: false,
          user: 'dev'
        }
      ]
    }

    list.mockResolvedValueOnce(sshRegistry)
    render(<ConnectionsRegistrySection />)

    await screen.findByText('Build host')
    fireEvent.click(screen.getByRole('button', { name: 'Edit' }))
    const pathInput = screen.getByPlaceholderText('auto-detect') as HTMLInputElement
    expect(pathInput.value).toBe('/opt/hermes/bin/hermes')
    fireEvent.change(pathInput, { target: { value: '   ' } })
    fireEvent.click(screen.getByText('Save connection').closest('button')!)

    await waitFor(() => expect(save).toHaveBeenCalledTimes(1))
    expect(save.mock.calls[0][0]).toMatchObject({ id: 'build-host', remoteHermesPath: '' })
  })

  it('offers every kind on create and disables Local while the managed entry exists', async () => {
    render(<ConnectionsRegistrySection />)

    await waitFor(() => expect(screen.getByText('Homelab')).toBeTruthy())
    fireEvent.click(screen.getByText('Add connection'))

    const localKind = screen.getByRole('button', { name: 'Local' }) as HTMLButtonElement
    expect(localKind.disabled).toBe(true)
    expect(screen.getByRole('button', { name: 'Hermes Cloud' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Remote gateway' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'SSH' })).toBeTruthy()
  })

  it('rejects a duplicate gateway URL in the save path with an inline error', async () => {
    render(<ConnectionsRegistrySection />)

    await waitFor(() => expect(screen.getByText('Homelab')).toBeTruthy())
    fireEvent.click(screen.getByText('Add connection'))

    fireEvent.change(screen.getByPlaceholderText('Homelab'), { target: { value: 'Homelab twin' } })
    // Same URL modulo case + trailing slash: normalized-dupe of the existing entry.
    fireEvent.change(screen.getByPlaceholderText('http://homelab.lan:9119'), {
      target: { value: 'HTTP://HOMELAB.LAN:9119/' }
    })
    fireEvent.click(screen.getByText('Save connection').closest('button')!)

    await waitFor(() =>
      expect(screen.getByText('A connection to this gateway URL already exists (“Homelab”).')).toBeTruthy()
    )
    expect(save).not.toHaveBeenCalled()
  })

  it('keeps the primary fallback configurable while last-used restore is enabled', async () => {
    list.mockResolvedValueOnce({ ...registry, launchMode: 'last-used' })
    render(<ConnectionsRegistrySection />)

    await waitFor(() => expect(screen.getByText('Homelab')).toBeTruthy())
    const makePrimary = screen.getByText('Make primary').closest('button')!

    expect(makePrimary.disabled).toBe(false)
    fireEvent.click(makePrimary)

    await waitFor(() => expect(setPrimary).toHaveBeenCalledWith('homelab'))
  })

  it('lets users opt into restoring the last-used source', async () => {
    render(<ConnectionsRegistrySection />)

    const launchSetting = await screen.findByText('At startup, return to Sessions on the last-used gateway')
    const addConnection = screen.getByText('Add connection')

    expect(addConnection.compareDocumentPosition(launchSetting) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    fireEvent.click(screen.getByRole('switch', { name: 'At startup, return to Sessions on the last-used gateway' }))

    await waitFor(() => expect(setLaunchMode).toHaveBeenCalledWith('last-used'))
  })

  it('offers the launch preference even for a single source', async () => {
    // A local-only registry is the drift state from #90174, and the launch
    // toggle is the control that lets a user out of it. Hiding it there left
    // hand-editing connections.json as the only recourse.
    list.mockResolvedValueOnce({ ...registry, connections: [registry.connections[0]] })

    render(<ConnectionsRegistrySection />)

    await waitFor(() => expect(list).toHaveBeenCalledTimes(1))
    expect(screen.getByText('At startup, return to Sessions on the last-used gateway')).toBeTruthy()
  })

  it('keeps search out of the way for a small registry', async () => {
    render(<ConnectionsRegistrySection />)

    await waitFor(() => expect(screen.getByText('Homelab')).toBeTruthy())
    expect(screen.queryByRole('searchbox', { name: 'Search gateways…' })).toBeNull()
  })

  it('sorts a large registry and searches names and endpoints', async () => {
    const largeRegistry: DesktopConnectionsRegistry = {
      ...registry,
      connections: [
        {
          authMode: 'token',
          id: 'zulu',
          kind: 'remote',
          label: 'Zulu',
          tokenPreview: null,
          tokenSet: false,
          url: 'https://zulu.example.test'
        },
        registry.connections[0],
        ...Array.from({ length: 6 }, (_, index) => ({
          authMode: 'token' as const,
          id: `gateway-${index}`,
          kind: 'remote' as const,
          label: index === 0 ? 'Alpha' : `Gateway ${index}`,
          tokenPreview: null,
          tokenSet: false,
          url:
            index === 4
              ? 'https://studio.example.test'
              : index === 5
                ? 'https://studio-archive.example.test'
                : `https://gateway-${index}.example.test`
        }))
      ]
    }

    list.mockResolvedValueOnce(largeRegistry)
    render(
      <div data-testid="settings-scroller" style={{ height: 400, overflowY: 'auto' }}>
        <ConnectionsRegistrySection />
      </div>
    )

    const search = await screen.findByRole('searchbox', { name: 'Search gateways…' })
    expect(search.parentElement?.className).toContain('mt-3')
    expect(search.parentElement?.className).toContain('mb-0')
    const settingsScroller = screen.getByTestId('settings-scroller')
    settingsScroller.scrollTop = 200
    vi.spyOn(search, 'getBoundingClientRect')
      .mockReturnValueOnce({
        bottom: 152,
        height: 32,
        left: 0,
        right: 0,
        top: 120,
        width: 0,
        x: 0,
        y: 120,
        toJSON: () => ({})
      })
      .mockReturnValueOnce({
        bottom: 152,
        height: 32,
        left: 0,
        right: 0,
        top: 120,
        width: 0,
        x: 0,
        y: 120,
        toJSON: () => ({})
      })
      .mockReturnValueOnce({
        bottom: 152,
        height: 32,
        left: 0,
        right: 0,
        top: 120,
        width: 0,
        x: 0,
        y: 120,
        toJSON: () => ({})
      })
      .mockReturnValue({
        bottom: 182,
        height: 32,
        left: 0,
        right: 0,
        top: 150,
        width: 0,
        x: 0,
        y: 150,
        toJSON: () => ({})
      })
    const alpha = screen.getByText('Alpha')
    const zulu = screen.getByText('Zulu')
    expect(alpha.compareDocumentPosition(zulu) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()

    fireEvent.change(search, { target: { value: 'studio' } })

    expect(settingsScroller.scrollTop).toBe(200)
    expect(screen.getByText('Gateway 4')).toBeTruthy()
    expect(screen.getByText('Gateway 5')).toBeTruthy()
    expect(screen.queryByText('Alpha')).toBeNull()

    settingsScroller.scrollTop = 260
    fireEvent.change(search, { target: { value: 'studio.example' } })
    expect(settingsScroller.scrollTop).toBe(290)
    expect(screen.getByText('Gateway 4')).toBeTruthy()
    expect(screen.queryByText('Gateway 5')).toBeNull()

    fireEvent.change(search, { target: { value: '' } })
    expect(search.closest<HTMLElement>('.border-t')?.style.minHeight).toBe('')
  })

  it('tests a connection through the bridge', async () => {
    render(<ConnectionsRegistrySection />)

    await waitFor(() => expect(screen.getByText('Homelab')).toBeTruthy())
    fireEvent.click(screen.getAllByText('Test')[0])

    await waitFor(() => expect(test).toHaveBeenCalled())
  })
})

describe('dedupe helpers', () => {
  it('normalizes gateway URLs (trim, trailing slashes, lowercase)', () => {
    expect(normalizeGatewayUrl(' HTTP://Homelab.LAN:9119// ')).toBe('http://homelab.lan:9119')
  })

  it('normalizes ssh composites and defaults the port', () => {
    expect(sshCompositeKey('alice@Box')).toBe('alice@box:22')
    expect(sshCompositeKey('alice@box:22')).toBe('alice@box:22')
    expect(sshCompositeKey('box:2222')).toBe('@box:2222')
    expect(sshCompositeKey('  ')).toBe('')
  })

  it('finds at most one local entry', () => {
    expect(
      findDuplicateConnection({ host: '', id: null, kind: 'local', remoteProfile: '', url: '' }, registry.connections)
    ).toMatchObject({ id: 'local' })
    // Editing the local entry itself is not a self-collision.
    expect(
      findDuplicateConnection(
        { host: '', id: 'local', kind: 'local', remoteProfile: '', url: '' },
        registry.connections
      )
    ).toBeNull()
  })

  it('keys remote/cloud dupes on the normalized URL across both kinds', () => {
    expect(
      findDuplicateConnection(
        { host: '', id: null, kind: 'cloud', remoteProfile: '', url: 'http://HOMELAB.lan:9119/' },
        registry.connections
      )
    ).toMatchObject({ id: 'homelab' })
    expect(
      findDuplicateConnection(
        { host: '', id: null, kind: 'remote', remoteProfile: '', url: 'http://other.lan:9119' },
        registry.connections
      )
    ).toBeNull()
    // Editing the entry itself is not a self-collision.
    expect(
      findDuplicateConnection(
        { host: '', id: 'homelab', kind: 'remote', remoteProfile: '', url: 'http://homelab.lan:9119' },
        registry.connections
      )
    ).toBeNull()
  })

  it('keys ssh dupes on user@host:port + remote profile', () => {
    const connections = [
      ...registry.connections,
      {
        host: 'box',
        id: 'box',
        kind: 'ssh' as const,
        label: 'Box',
        port: 22,
        remoteProfile: 'work',
        tokenPreview: null,
        tokenSet: false,
        user: 'alice'
      }
    ]

    expect(
      findDuplicateConnection(
        { host: 'alice@box:22', id: null, kind: 'ssh', remoteProfile: 'work', url: '' },
        connections
      )
    ).toMatchObject({ id: 'box' })
    // Different profile on the same host is a distinct agent source.
    expect(
      findDuplicateConnection(
        { host: 'alice@box:22', id: null, kind: 'ssh', remoteProfile: 'other', url: '' },
        connections
      )
    ).toBeNull()
  })

  it('hints "Same backend as" only on later rows sharing an install_id', () => {
    const spark = { id: 'spark', installId: 'aaa', label: 'Spark' }
    const sparkTs = { id: 'spark-ts', installId: 'aaa', label: 'Spark TS' }
    const mini = { id: 'mini', installId: 'bbb', label: 'Mini' }
    const legacy = { id: 'old', label: 'Old box' }
    const connections = [spark, sparkTs, mini, legacy]

    // The first occurrence carries no hint; the later duplicate points back.
    expect(sameBackendPeerLabel(spark, connections)).toBeNull()
    expect(sameBackendPeerLabel(sparkTs, connections)).toBe('Spark')
    // Unique ids and id-less (older backend) rows never hint.
    expect(sameBackendPeerLabel(mini, connections)).toBeNull()
    expect(sameBackendPeerLabel(legacy, connections)).toBeNull()
  })
})

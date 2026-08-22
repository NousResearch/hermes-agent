import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { DesktopConnectionsRegistry } from '@/global'
import { $connection } from '@/store/session'

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

  it('keeps the launch preference out of the way for a single source', async () => {
    list.mockResolvedValueOnce({ ...registry, connections: [registry.connections[0]] })

    render(<ConnectionsRegistrySection />)

    await waitFor(() => expect(list).toHaveBeenCalledTimes(1))
    expect(screen.queryByText('At startup, return to Sessions on the last-used gateway')).toBeNull()
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

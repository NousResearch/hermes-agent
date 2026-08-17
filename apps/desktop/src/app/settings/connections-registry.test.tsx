import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { DesktopConnectionsRegistry } from '@/global'

import { ConnectionsRegistrySection, findDuplicateConnection, normalizeGatewayUrl, sshCompositeKey } from './connections-registry'

const list = vi.fn()
const save = vi.fn()
const remove = vi.fn()
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
  list.mockResolvedValue(registry)
  save.mockResolvedValue({ connection: registry.connections[1], ok: true, registry })
  remove.mockResolvedValue({ ok: true, registry: { ...registry, connections: [registry.connections[0]] } })
  setPrimary.mockResolvedValue({ ok: true, registry: { ...registry, primary: 'homelab' } })
  test.mockResolvedValue({ ok: true, reachable: true })
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { connections: { list, remove, save, setPrimary, test } }
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('ConnectionsRegistrySection', () => {
  it('lists registered connections with primary + local pills', async () => {
    render(<ConnectionsRegistrySection />)

    await waitFor(() => expect(screen.getByText('Homelab')).toBeTruthy())
    // Label and the managed pill share the copy, so expect both instances.
    expect(screen.getAllByText('This device').length).toBeGreaterThan(0)
    expect(screen.getByText('Primary')).toBeTruthy()
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

  it('makes a non-primary connection primary', async () => {
    render(<ConnectionsRegistrySection />)

    await waitFor(() => expect(screen.getByText('Homelab')).toBeTruthy())
    fireEvent.click(screen.getByText('Make primary'))

    await waitFor(() => expect(setPrimary).toHaveBeenCalledWith('homelab'))
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
})

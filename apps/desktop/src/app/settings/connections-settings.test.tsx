import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { DesktopConnectionsRegistry } from '@/global'

import { ConnectionsSettings } from './connections-settings'

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

describe('ConnectionsSettings', () => {
  it('lists registered connections with primary + local pills', async () => {
    render(<ConnectionsSettings />)

    await waitFor(() => expect(screen.getByText('Homelab')).toBeTruthy())
    // Label and the managed pill share the copy, so expect both instances.
    expect(screen.getAllByText('This device').length).toBeGreaterThan(0)
    expect(screen.getByText('Primary')).toBeTruthy()
    expect(list).toHaveBeenCalledTimes(1)
  })

  it('opens the add-connection editor and saves with a required label', async () => {
    render(<ConnectionsSettings />)

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

  it('makes a non-primary connection primary', async () => {
    render(<ConnectionsSettings />)

    await waitFor(() => expect(screen.getByText('Homelab')).toBeTruthy())
    fireEvent.click(screen.getByText('Make primary'))

    await waitFor(() => expect(setPrimary).toHaveBeenCalledWith('homelab'))
  })

  it('tests a connection through the bridge', async () => {
    render(<ConnectionsSettings />)

    await waitFor(() => expect(screen.getByText('Homelab')).toBeTruthy())
    fireEvent.click(screen.getAllByText('Test')[0])

    await waitFor(() => expect(test).toHaveBeenCalled())
  })
})

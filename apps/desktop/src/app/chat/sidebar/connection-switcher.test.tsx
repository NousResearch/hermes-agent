import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { DesktopConnectionsRegistry, DesktopProfileRoute, DesktopRegistryConnection } from '@/global'
import { $findInPage } from '@/store/find-in-page'

import { ConnectionSwitcher } from './connection-switcher'

// Radix menus use pointer capture; jsdom does not implement it.
Element.prototype.hasPointerCapture ??= () => false
Element.prototype.setPointerCapture ??= () => undefined
Element.prototype.releasePointerCapture ??= () => undefined
Element.prototype.scrollIntoView ??= () => undefined
globalThis.ResizeObserver ??= class ResizeObserver {
  disconnect() {}
  observe() {}
  unobserve() {}
}

vi.mock('@/store/connections', () => ({
  $activeConnectionId: atom<null | string>('local'),
  $connectionsRegistry: atom<DesktopConnectionsRegistry | null>(null),
  $pendingConnectionId: atom<null | string>(null),
  selectConnection: vi.fn(async () => undefined)
}))

vi.mock('@/store/default-profile', () => ({
  $defaultProfileRoute: atom<DesktopProfileRoute | null>(null)
}))

vi.mock('@/store/windows', () => ({
  canOpenNewWindow: vi.fn(() => true),
  openNewWindow: vi.fn(async () => undefined)
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      profiles: {
        openInNewWindow: 'Open in new window',
        openInNewWindowHint: 'Shift-click a gateway to open it in a new window',
        switchConnectionFailed: (name: string) => `Could not connect to ${name}`,
        switchToConnection: (name: string) => `Switch to ${name}`,
        connectGateway: 'Manage gateways…'
      },
      settings: {
        connections: {
          noSearchResults: 'No gateways match your search.',
          searchPlaceholder: 'Search gateways…',
          kindCloud: 'Hermes Cloud',
          kindLocal: 'Local',
          kindRemote: 'Remote gateway',
          kindSsh: 'SSH',
          title: 'Registered gateways'
        }
      }
    }
  })
}))

const connectionStore = await import('@/store/connections')
const $activeConnectionId = connectionStore.$activeConnectionId as ReturnType<typeof atom<null | string>>
const $connectionsRegistry = connectionStore.$connectionsRegistry
const $pendingConnectionId = connectionStore.$pendingConnectionId
const selectConnection = vi.mocked(connectionStore.selectConnection)
const windowStore = await import('@/store/windows')
const canOpenNewWindow = vi.mocked(windowStore.canOpenNewWindow)
const openNewWindow = vi.mocked(windowStore.openNewWindow)

const defaultProfileStore = await import('@/store/default-profile')

const $defaultProfileRoute = defaultProfileStore.$defaultProfileRoute as ReturnType<
  typeof atom<DesktopProfileRoute | null>
>

const onConnect = vi.fn()

const connection = (
  id: string,
  label: string,
  kind: 'local' | 'remote' = 'remote',
  extra: Partial<DesktopRegistryConnection> = {}
) => ({
  id,
  kind,
  label,
  tokenPreview: null,
  tokenSet: false,
  ...extra
})

const registry = (connections: ReturnType<typeof connection>[]): DesktopConnectionsRegistry => ({
  connections,
  primary: connections[0]?.id ?? 'local',
  secureTokenStorage: true,
  version: 2
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  canOpenNewWindow.mockReturnValue(true)
  $connectionsRegistry.set(null)
  $defaultProfileRoute.set(null)
  $activeConnectionId.set('local')
  $pendingConnectionId.set(null)
  $findInPage.set({ active: false, query: '', matchOrdinal: 0, matchCount: 0, focusRequest: 0 })
})

describe('ConnectionSwitcher', () => {
  it('adds no source chrome for a local-only setup', () => {
    $connectionsRegistry.set(registry([connection('local', 'This device', 'local')]))
    render(<ConnectionSwitcher onConnect={onConnect} />)

    expect(screen.queryByRole('group', { name: 'Registered gateways' })).toBeNull()
  })

  it('shows a named source selector instead of profile-like gateway glyphs', () => {
    $connectionsRegistry.set(
      registry([
        connection('local', 'This device', 'local'),
        connection('homelab', 'Homelab'),
        connection('work-vps', 'Work VPS')
      ])
    )
    render(<ConnectionSwitcher onConnect={onConnect} />)

    const trigger = screen.getByRole('button', { name: 'Registered gateways: This device' })

    expect(trigger.textContent).toContain('This device')

    fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.click(screen.getByRole('menuitemradio', { name: 'Homelab' }))
    expect(selectConnection).toHaveBeenCalledWith('homelab')

    fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.click(screen.getByRole('menuitem', { name: 'Manage gateways…' }))
    expect(onConnect).toHaveBeenCalledTimes(1)
    expect(selectConnection).toHaveBeenCalledTimes(1)
  })

  it('keeps source controls stable while a remote is opening', () => {
    $connectionsRegistry.set(registry([connection('local', 'This device', 'local'), connection('homelab', 'Homelab')]))
    $pendingConnectionId.set('homelab')
    render(<ConnectionSwitcher onConnect={onConnect} />)

    expect(screen.getByRole('group', { name: 'Registered gateways' }).getAttribute('aria-busy')).toBe('true')
  })

  it('keeps small gateway lists simple and naturally sorted', () => {
    $connectionsRegistry.set(
      registry([
        connection('zulu', 'Zulu'),
        connection('local', 'This device', 'local'),
        connection('studio-10', 'Studio 10'),
        connection('alpha', 'alpha'),
        connection('studio-2', 'Studio 2')
      ])
    )
    render(<ConnectionSwitcher onConnect={onConnect} />)

    const trigger = screen.getByRole('button', { name: 'Registered gateways: This device' })
    fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })

    expect(screen.queryByPlaceholderText('Search gateways…')).toBeNull()
    expect(screen.getAllByRole('menuitemradio').map(item => item.textContent)).toEqual([
      'This device',
      'alpha',
      'Studio 2',
      'Studio 10',
      'Zulu'
    ])
  })

  it('adds search at eight gateways and filters stable results without moving the connect action', async () => {
    $connectionsRegistry.set(
      registry([
        connection('zulu', 'Zulu'),
        connection('local', 'This device', 'local'),
        connection('studio-10', 'Studio 10'),
        connection('alpha', 'Alpha'),
        connection('studio-2', 'Studio 2'),
        connection('work', 'Work VPS'),
        connection('homelab', 'Homelab'),
        connection('cloud', 'Cloud lab')
      ])
    )
    render(<ConnectionSwitcher onConnect={onConnect} />)

    const trigger = screen.getByRole('button', { name: 'Registered gateways: This device' })
    fireEvent.pointerDown(trigger, {
      button: 0,
      pointerType: 'mouse'
    })

    const search = screen.getByPlaceholderText('Search gateways…')
    expect(screen.getByRole('menuitem', { name: 'Manage gateways…' })).toBeTruthy()
    expect(screen.getAllByRole('menuitemradio').map(item => item.textContent)).toEqual([
      'This device',
      'Alpha',
      'Cloud lab',
      'Homelab',
      'Studio 2',
      'Studio 10',
      'Work VPS',
      'Zulu'
    ])

    fireEvent.change(search, { target: { value: 'studio 10' } })
    expect(screen.getAllByRole('menuitemradio').map(item => item.textContent)).toEqual(['Studio 10'])

    const result = screen.getByRole('menuitemradio', { name: 'Studio 10' })
    fireEvent.keyDown(search, { key: 'ArrowDown' })
    expect(globalThis.document.activeElement).toBe(result)

    $findInPage.set({ active: true, query: '', matchOrdinal: 0, matchCount: 0, focusRequest: 0 })
    result.focus()
    fireEvent.keyDown(result, { key: 'f', metaKey: true })
    expect(globalThis.document.activeElement).toBe(search)
    expect($findInPage.get().active).toBe(false)

    fireEvent.keyDown(search, { key: 'Escape' })
    expect(screen.queryByPlaceholderText('Search gateways…')).toBeNull()
    await waitFor(() => expect(globalThis.document.activeElement).toBe(trigger))

    fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
    expect((screen.getByPlaceholderText('Search gateways…') as HTMLInputElement).value).toBe('')

    fireEvent.click(screen.getByRole('menuitemradio', { name: 'Studio 10' }))
    expect(selectConnection).toHaveBeenCalledWith('studio-10')

    fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
    expect((screen.getByPlaceholderText('Search gateways…') as HTMLInputElement).value).toBe('')
  })

  it('explains an empty large-list search', () => {
    $connectionsRegistry.set(
      registry([
        connection('local', 'This device', 'local'),
        ...Array.from({ length: 7 }, (_, index) => connection(`remote-${index}`, `Remote ${index}`))
      ])
    )
    render(<ConnectionSwitcher onConnect={onConnect} />)

    fireEvent.pointerDown(screen.getByRole('button', { name: 'Registered gateways: This device' }), {
      button: 0,
      pointerType: 'mouse'
    })
    fireEvent.change(screen.getByPlaceholderText('Search gateways…'), { target: { value: 'missing' } })

    expect(screen.getByText('No gateways match your search.')).toBeTruthy()
    expect(screen.getByRole('menuitem', { name: 'Manage gateways…' })).toBeTruthy()
  })

  // The window lifecycle owns IPC; this consumer paints its published cache.
  it('repaints the menu after the registry changes without reload', () => {
    const before = registry([connection('local', 'This device', 'local'), connection('homelab', 'Homelab')])

    const after = registry([
      connection('local', 'This device', 'local'),
      connection('homelab', 'Homelab'),
      connection('w2-probe', 'W2Probe')
    ])

    $connectionsRegistry.set(before)
    render(<ConnectionSwitcher onConnect={onConnect} />)

    const trigger = screen.getByRole('button', { name: 'Registered gateways: This device' })

    fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
    expect(screen.queryByRole('menuitemradio', { name: 'W2Probe' })).toBeNull()
    fireEvent.keyDown(document, { key: 'Escape' })

    act(() => $connectionsRegistry.set(after))

    fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
    expect(screen.getByRole('menuitemradio', { name: 'W2Probe' })).toBeTruthy()
  })

  // Shift is the desktop convention for "open this somewhere else". The radio
  // group cannot carry it — re-picking the active row fires no value change —
  // so each row handles the modified press itself.
  it('shift-clicks a gateway into a peer window and leaves this window alone', () => {
    $connectionsRegistry.set(registry([connection('local', 'This device', 'local'), connection('homelab', 'Homelab')]))
    render(<ConnectionSwitcher onConnect={onConnect} />)

    fireEvent.pointerDown(screen.getByRole('button', { name: 'Registered gateways: This device' }), {
      button: 0,
      pointerType: 'mouse'
    })
    fireEvent.pointerDown(screen.getByRole('menuitemradio', { name: 'Homelab' }), {
      button: 0,
      pointerType: 'mouse',
      shiftKey: true
    })

    expect(openNewWindow).toHaveBeenCalledWith({ connectionId: 'homelab', profile: 'default' })
    expect(selectConnection).not.toHaveBeenCalled()
  })

  it('opens a peer window from the gateway that is already active', () => {
    $connectionsRegistry.set(registry([connection('local', 'This device', 'local'), connection('homelab', 'Homelab')]))
    $activeConnectionId.set('homelab')
    render(<ConnectionSwitcher onConnect={onConnect} />)
    fireEvent.click(screen.getByRole('button', { name: 'Registered gateways: Homelab' }))
    fireEvent.pointerDown(screen.getByRole('button', { name: 'Registered gateways: Homelab' }), {
      button: 0,
      pointerType: 'mouse'
    })
    fireEvent.pointerDown(screen.getByRole('menuitemradio', { name: 'Homelab' }), {
      button: 0,
      pointerType: 'mouse',
      shiftKey: true
    })

    expect(openNewWindow).toHaveBeenCalledWith({ connectionId: 'homelab', profile: 'default' })
    expect(selectConnection).not.toHaveBeenCalled()
  })

  it('launches the profile this app defaults to on that gateway', () => {
    $connectionsRegistry.set(registry([connection('local', 'This device', 'local'), connection('homelab', 'Homelab')]))
    $defaultProfileRoute.set({ connectionId: 'homelab', profile: 'research' })
    render(<ConnectionSwitcher onConnect={onConnect} />)

    fireEvent.pointerDown(screen.getByRole('button', { name: 'Registered gateways: This device' }), {
      button: 0,
      pointerType: 'mouse'
    })
    fireEvent.pointerDown(screen.getByRole('menuitemradio', { name: 'Homelab' }), {
      button: 0,
      pointerType: 'mouse',
      shiftKey: true
    })

    expect(openNewWindow).toHaveBeenCalledWith({ connectionId: 'homelab', profile: 'research' })
  })

  it('falls back to the SSH pinned profile when no default names that gateway', () => {
    $connectionsRegistry.set(
      registry([
        connection('local', 'This device', 'local'),
        connection('vps', 'Work VPS', 'remote', { remoteProfile: 'ops' })
      ])
    )
    render(<ConnectionSwitcher onConnect={onConnect} />)

    fireEvent.pointerDown(screen.getByRole('button', { name: 'Registered gateways: This device' }), {
      button: 0,
      pointerType: 'mouse'
    })
    fireEvent.pointerDown(screen.getByRole('menuitemradio', { name: 'Work VPS' }), {
      button: 0,
      pointerType: 'mouse',
      shiftKey: true
    })

    expect(openNewWindow).toHaveBeenCalledWith({ connectionId: 'vps', profile: 'ops' })
  })

  it('opens a peer window from the keyboard with shift+enter', () => {
    $connectionsRegistry.set(registry([connection('local', 'This device', 'local'), connection('homelab', 'Homelab')]))
    render(<ConnectionSwitcher onConnect={onConnect} />)

    fireEvent.pointerDown(screen.getByRole('button', { name: 'Registered gateways: This device' }), {
      button: 0,
      pointerType: 'mouse'
    })
    fireEvent.keyDown(screen.getByRole('menuitemradio', { name: 'Homelab' }), { key: 'Enter', shiftKey: true })

    expect(openNewWindow).toHaveBeenCalledWith({ connectionId: 'homelab', profile: 'default' })
    expect(selectConnection).not.toHaveBeenCalled()
  })

  it('does not advertise the peer-window gesture where no window can open', () => {
    canOpenNewWindow.mockReturnValue(false)
    $connectionsRegistry.set(registry([connection('local', 'This device', 'local'), connection('homelab', 'Homelab')]))
    render(<ConnectionSwitcher onConnect={onConnect} />)

    fireEvent.pointerDown(screen.getByRole('button', { name: 'Registered gateways: This device' }), {
      button: 0,
      pointerType: 'mouse'
    })

    expect(screen.queryByText('Shift-click a gateway to open it in a new window')).toBeNull()

    fireEvent.pointerDown(screen.getByRole('menuitemradio', { name: 'Homelab' }), {
      button: 0,
      pointerType: 'mouse',
      shiftKey: true
    })
    expect(openNewWindow).not.toHaveBeenCalled()
  })
})

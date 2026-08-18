import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { DesktopConnectionsRegistry } from '@/global'
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
  initializeConnectionsRegistry: vi.fn(async () => null),
  refreshConnectionsRegistry: vi.fn(async () => null),
  selectConnection: vi.fn(async () => undefined)
}))

vi.mock('@/store/boot', () => ({
  $desktopBoot: atom({
    error: null,
    fakeMode: false,
    message: 'Starting',
    phase: 'renderer.init',
    progress: 2,
    running: true,
    timestamp: 0,
    visible: true
  })
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      profiles: {
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
const bootStore = await import('@/store/boot')
const $activeConnectionId = connectionStore.$activeConnectionId as ReturnType<typeof atom<null | string>>
const $connectionsRegistry = connectionStore.$connectionsRegistry
const $desktopBoot = bootStore.$desktopBoot
const $pendingConnectionId = connectionStore.$pendingConnectionId
const initializeConnectionsRegistry = vi.mocked(connectionStore.initializeConnectionsRegistry)
const refreshConnectionsRegistry = vi.mocked(connectionStore.refreshConnectionsRegistry)
const selectConnection = vi.mocked(connectionStore.selectConnection)
const onConnect = vi.fn()

const connection = (id: string, label: string, kind: 'local' | 'remote' = 'remote') => ({
  id,
  kind,
  label,
  tokenPreview: null,
  tokenSet: false
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
  $connectionsRegistry.set(null)
  $activeConnectionId.set('local')
  $desktopBoot.set({
    error: null,
    fakeMode: false,
    message: 'Starting',
    phase: 'renderer.init',
    progress: 2,
    running: true,
    timestamp: 0,
    visible: true
  })
  $pendingConnectionId.set(null)
  $findInPage.set({ active: false, query: '', matchOrdinal: 0, matchCount: 0 })
})

describe('ConnectionSwitcher', () => {
  it('waits for primary boot fetches before restoring the launch source', async () => {
    $connectionsRegistry.set(registry([connection('local', 'This device', 'local'), connection('homelab', 'Homelab')]))
    render(<ConnectionSwitcher onConnect={onConnect} />)

    expect(refreshConnectionsRegistry).toHaveBeenCalledTimes(1)
    expect(initializeConnectionsRegistry).not.toHaveBeenCalled()

    $desktopBoot.set({
      ...$desktopBoot.get(),
      phase: 'renderer.ready',
      progress: 100,
      running: false,
      visible: false
    })

    await waitFor(() => expect(initializeConnectionsRegistry).toHaveBeenCalledTimes(1))
  })

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
    expect(trigger.getAttribute('data-variant')).toBe('ghost')
    expect(trigger.querySelector('[data-connection-kind="local"] svg')).toBeTruthy()
    expect(trigger.querySelector('.codicon-home')).toBeNull()

    fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.click(screen.getByRole('menuitemradio', { name: 'Homelab' }))
    expect(selectConnection).toHaveBeenCalledWith('homelab')

    fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
    fireEvent.click(screen.getByRole('menuitem', { name: 'Manage gateways…' }))
    expect(onConnect).toHaveBeenCalledTimes(1)
    expect(selectConnection).toHaveBeenCalledTimes(1)
  })

  it('fits the shared statusbar slot without changing its gateway identity', () => {
    $connectionsRegistry.set(registry([connection('local', 'This device', 'local'), connection('homelab', 'Homelab')]))
    render(<ConnectionSwitcher compact onConnect={onConnect} />)

    const group = screen.getByRole('group', { name: 'Registered gateways' })
    const trigger = screen.getByRole('button', { name: 'Registered gateways: This device' })

    expect(group.className).toContain('h-full')
    expect(group.className).toContain('min-w-20')
    expect(group.className).toContain('max-w-40')
    expect(group.className).toContain('shrink')
    expect(group.className).toContain('overflow-hidden')
    expect(trigger.className).toContain('text-[0.6875rem]')
    expect(trigger.className).toContain('min-w-0')
    expect(trigger.className).toContain('overflow-hidden')
    expect(trigger.textContent).toContain('This device')
  })

  it('keeps source controls stable while a remote is opening', () => {
    $connectionsRegistry.set(registry([connection('local', 'This device', 'local'), connection('homelab', 'Homelab')]))
    $pendingConnectionId.set('homelab')
    render(<ConnectionSwitcher onConnect={onConnect} />)

    expect(screen.getByRole('group', { name: 'Registered gateways' }).getAttribute('aria-busy')).toBe('true')
    expect(
      screen.getByRole('button', { name: 'Registered gateways: This device' }).querySelector('.animate-spin')
    ).toBeTruthy()
  })

  it.each([2, 20])('uses the same stable source selector for %i registered backends', count => {
    $connectionsRegistry.set(
      registry([
        connection('local', 'This device', 'local'),
        ...Array.from({ length: count - 1 }, (_, index) => connection(`remote-${index}`, `Remote ${index}`))
      ])
    )
    render(<ConnectionSwitcher onConnect={onConnect} />)

    expect(screen.getByRole('button', { name: 'Registered gateways: This device' })).toBeTruthy()
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

    $findInPage.set({ active: true, query: '', matchOrdinal: 0, matchCount: 0 })
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
    expect(globalThis.document.querySelector('[data-slot="dropdown-menu-radio-group"]')?.className).toContain('h-48')
  })

  it('announces a pending switch in the compact source menu', () => {
    $connectionsRegistry.set(
      registry([
        connection('local', 'This device', 'local'),
        ...Array.from({ length: 6 }, (_, index) => connection(`remote-${index}`, `Remote ${index}`))
      ])
    )
    $pendingConnectionId.set('remote-3')
    render(<ConnectionSwitcher onConnect={onConnect} />)

    expect(screen.getByRole('group', { name: 'Registered gateways' }).getAttribute('aria-busy')).toBe('true')
  })
})

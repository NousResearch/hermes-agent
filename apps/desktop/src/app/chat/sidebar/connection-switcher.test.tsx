import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { DesktopConnectionsRegistry } from '@/global'

import { ConnectionSwitcher } from './connection-switcher'

// Radix menus use pointer capture; jsdom does not implement it.
Element.prototype.hasPointerCapture ??= () => false
Element.prototype.setPointerCapture ??= () => undefined
Element.prototype.releasePointerCapture ??= () => undefined

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
        connectGateway: 'Connect another Hermes gateway…'
      },
      settings: { connections: { title: 'Registered gateways' } }
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
    fireEvent.click(screen.getByRole('menuitem', { name: 'Connect another Hermes gateway…' }))
    expect(onConnect).toHaveBeenCalledTimes(1)
    expect(selectConnection).toHaveBeenCalledTimes(1)
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

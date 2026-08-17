import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { DesktopConnectionsRegistry } from '@/global'

import { ConnectionSwitcher } from './connection-switcher'

vi.mock('@/store/connections', () => ({
  $activeConnectionId: atom<null | string>('local'),
  $connectionsRegistry: atom<DesktopConnectionsRegistry | null>(null),
  $pendingConnectionId: atom<null | string>(null),
  refreshConnectionsRegistry: vi.fn(async () => null),
  selectConnection: vi.fn(async () => undefined)
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      profiles: {
        switchConnectionFailed: (name: string) => `Could not connect to ${name}`,
        switchToConnection: (name: string) => `Switch to ${name}`
      },
      settings: { connections: { title: 'Connections' } }
    }
  })
}))

const connectionStore = await import('@/store/connections')
const $activeConnectionId = connectionStore.$activeConnectionId as ReturnType<typeof atom<null | string>>
const $connectionsRegistry = connectionStore.$connectionsRegistry
const $pendingConnectionId = connectionStore.$pendingConnectionId
const selectConnection = vi.mocked(connectionStore.selectConnection)

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
  $pendingConnectionId.set(null)
})

describe('ConnectionSwitcher', () => {
  it('adds no source chrome for a local-only setup', () => {
    $connectionsRegistry.set(registry([connection('local', 'This device', 'local')]))
    render(<ConnectionSwitcher />)

    expect(screen.queryByRole('group', { name: 'Connections' })).toBeNull()
  })

  it('shows direct local and remote source controls for a small registry', () => {
    $connectionsRegistry.set(
      registry([
        connection('local', 'This device', 'local'),
        connection('homelab', 'Homelab'),
        connection('work-vps', 'Work VPS')
      ])
    )
    render(<ConnectionSwitcher />)

    expect(screen.getByRole('button', { name: 'Switch to This device' }).getAttribute('aria-pressed')).toBe('true')
    fireEvent.click(screen.getByRole('button', { name: 'Switch to Homelab' }))
    expect(selectConnection).toHaveBeenCalledWith('homelab')
  })

  it('keeps source controls stable while a remote is opening', () => {
    $connectionsRegistry.set(registry([connection('local', 'This device', 'local'), connection('homelab', 'Homelab')]))
    $pendingConnectionId.set('homelab')
    render(<ConnectionSwitcher />)

    expect(screen.getByRole('button', { name: 'Switch to Homelab' }).getAttribute('aria-busy')).toBe('true')
    expect(screen.getByRole('button', { name: 'Switch to This device' }).getAttribute('aria-busy')).toBe('false')
  })

  it('collapses a larger registry into one compact source select', () => {
    $connectionsRegistry.set(
      registry([
        connection('local', 'This device', 'local'),
        ...Array.from({ length: 6 }, (_, index) => connection(`remote-${index}`, `Remote ${index}`))
      ])
    )
    render(<ConnectionSwitcher />)

    expect(screen.getByRole('combobox', { name: 'Connections' })).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Switch to Homelab' })).toBeNull()
  })

  it('announces a pending switch in the compact source menu', () => {
    $connectionsRegistry.set(
      registry([
        connection('local', 'This device', 'local'),
        ...Array.from({ length: 6 }, (_, index) => connection(`remote-${index}`, `Remote ${index}`))
      ])
    )
    $pendingConnectionId.set('remote-3')
    render(<ConnectionSwitcher />)

    expect(screen.getByRole('group', { name: 'Connections' }).getAttribute('aria-busy')).toBe('true')
  })
})

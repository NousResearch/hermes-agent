// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as MenuBarTransparencyModule from '@/store/menu-bar-transparency'

const stores = vi.hoisted(() => ({ activeProfile: null as any, menuBarTransparency: null as any }))

vi.mock('@/themes', () => ({
  useTheme: () => ({ mode: 'dark', resolvedMode: 'dark', themeName: 'nous' })
}))

vi.mock('@/store/profile', async () => {
  const { atom } = await import('nanostores')
  stores.activeProfile = atom('default')

  return { $activeGatewayProfile: stores.activeProfile }
})

vi.mock('@/store/menu-bar-transparency', async importOriginal => {
  const actual = await importOriginal<typeof MenuBarTransparencyModule>()
  const { atom } = await import('nanostores')
  stores.menuBarTransparency = atom(20)

  return { ...actual, $menuBarTransparency: stores.menuBarTransparency }
})

vi.mock('@/lib/desktop-state-sync', () => ({
  onDesktopStateSync: () => () => {},
  requestActiveDesktopProfile: vi.fn()
}))

vi.mock('./tabs/appearance-tab', () => ({ AppearanceTab: () => <div>Appearance controls</div> }))
vi.mock('./tabs/atlas-tab', () => ({ AtlasTab: () => <div>Atlas controls</div> }))
vi.mock('./tabs/commands-tab', () => ({ CommandsTab: () => <div>Command controls</div> }))
vi.mock('./tabs/quick-tab', () => ({ QuickTab: () => <div>Quick controls</div> }))

import { MenuBarCompanionShell } from './shell'

beforeEach(() => {
  Object.defineProperty(window, 'matchMedia', {
    configurable: true,
    value: vi.fn(() => ({ matches: true }))
  })
})

afterEach(() => cleanup())

describe('MenuBarCompanionShell', () => {
  it('labels the shared settings tab Appearance and applies live glass variables', async () => {
    const { container } = render(<MenuBarCompanionShell />)
    const shell = container.querySelector<HTMLElement>('.hermes-menubar-shell')

    expect(shell?.style.getPropertyValue('--mbc-bg-alpha')).toBe('0.86')
    expect(screen.queryByRole('tab', { name: 'Status' })).toBeNull()

    fireEvent.click(screen.getByRole('tab', { name: 'Appearance' }))
    expect(screen.getByText('Appearance controls')).toBeTruthy()

    stores.menuBarTransparency.set(100)
    await waitFor(() => expect(shell?.style.getPropertyValue('--mbc-bg-alpha')).toBe('0.38'))
  })
})

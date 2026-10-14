/**
 * Reopening Settings lands on the tab last used. The tab lives in the route's
 * `?tab=`, so without memory a reopen snaps back to the default view; with
 * memory it reopens where it was left. An explicit `?tab=` still wins.
 */
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { resetViewMemory } from '@/store/view-memory'

vi.mock('./about-settings', () => ({ AboutSettings: () => null }))
vi.mock('./appearance-settings', () => ({ AppearanceSettings: () => null }))
vi.mock('./billing', () => ({ BILLING_VIEWS: ['overview', 'plans'], BillingSettings: () => null }))
vi.mock('./billing/use-billing-state', () => ({
  deriveBillingView: () => ({ status: 'normal', plan: undefined }),
  useBillingState: () => ({ data: undefined }),
  useSubscriptionState: () => ({ data: undefined })
}))
vi.mock('./config-settings', () => ({ ConfigSettings: () => <div data-testid="view-config" /> }))
vi.mock('./gateway-settings', () => ({ GatewaySettings: () => null }))
vi.mock('./keybind-settings', () => ({ KeybindSettings: () => null }))
vi.mock('./keys-settings', () => ({ KEYS_VIEWS: ['tools', 'settings'], KeysSettings: () => null }))
vi.mock('./notifications-settings', () => ({ NotificationsSettings: () => <div data-testid="view-notifications" /> }))
vi.mock('./plugin-settings', () => ({
  PAGE_SCOPED_PARAMS: [],
  PLUGINS_NAV_ICON: () => null,
  PluginSettingsPane: () => null,
  pluginSettingsNavChildren: () => [],
  usePluginSettingsRoute: () => ({ entries: [], missing: [], open: vi.fn(), pending: false, target: null })
}))
vi.mock('./providers-settings', () => ({
  PROVIDER_VIEWS: ['accounts', 'keys', 'custom-endpoints'],
  ProvidersSettings: () => null
}))
vi.mock('./sessions-settings', () => ({ SessionsSettings: () => null }))
vi.mock('./subpage-navigation', () => ({ SettingsSubpageHeader: () => null }))
vi.mock('./vault-settings', () => ({ VaultSettings: () => null, vaultOwnerKey: () => 'k' }))

const { SettingsView } = await import('./index')

function renderAt(pathname: string) {
  return render(
    <MemoryRouter initialEntries={[pathname]}>
      <SettingsView onClose={() => {}} />
    </MemoryRouter>
  )
}

const navRow = (container: HTMLElement, id: string) =>
  container.querySelector<HTMLElement>(`[data-tour="nav-${id}"]`)

beforeEach(resetViewMemory)

afterEach(cleanup)

describe('Settings remembers its last tab', () => {
  it('reopens on the tab last used', async () => {
    const first = renderAt('/settings')

    // Default: the Models config page.
    expect(screen.getByTestId('view-config')).toBeTruthy()

    await act(async () => {
      fireEvent.click(navRow(first.container, 'notifications')!)
    })
    expect(screen.getByTestId('view-notifications')).toBeTruthy()

    cleanup()

    // Reopen with no `?tab=` — memory decides.
    renderAt('/settings')
    expect(screen.getByTestId('view-notifications')).toBeTruthy()
  })

  it('lets an explicit ?tab= win over the remembered tab', async () => {
    const first = renderAt('/settings')

    await act(async () => {
      fireEvent.click(navRow(first.container, 'notifications')!)
    })

    cleanup()

    renderAt('/settings?tab=config:model')
    expect(screen.getByTestId('view-config')).toBeTruthy()
    expect(screen.queryByTestId('view-notifications')).toBeNull()
  })
})

describe('Settings remembers the sub-pane it was left on', () => {
  const current = (container: HTMLElement) =>
    container.querySelector<HTMLElement>('[aria-current="page"]')?.getAttribute('data-tour')

  const subRow = (container: HTMLElement, id: string) => container.querySelector<HTMLElement>(`[data-tour="nav-${id}"]`)

  it('reopens on the sub-pane last used inside a section', async () => {
    const first = renderAt('/settings?tab=notifications')

    await act(async () => {
      fireEvent.click(subRow(first.container, 'notifications:sounds')!)
    })

    expect(current(first.container)).toBe('nav-notifications:sounds')

    cleanup()

    // No `?tab=` and no `?page=`: memory decides both.
    const reopened = renderAt('/settings')

    expect(screen.getByTestId('view-notifications')).toBeTruthy()
    expect(current(reopened.container)).toBe('nav-notifications:sounds')
  })

  it('lets the route name a sub-pane and remembers that one instead', async () => {
    renderAt('/settings?tab=notifications&page=alerts')

    cleanup()

    const reopened = renderAt('/settings')

    expect(current(reopened.container)).toBe('nav-notifications:alerts')
    // ...and having named it, that is what the next bare open lands on.
    expect(current(reopened.container)).not.toBe('nav-notifications:sounds')
  })

  it('remembers a section whose sub-nav lives in its own param', async () => {
    const first = renderAt('/settings?tab=providers')

    await act(async () => {
      fireEvent.click(subRow(first.container, 'pview:keys')!)
    })

    cleanup()

    const reopened = renderAt('/settings')

    await act(async () => {})

    expect(current(reopened.container)).toBe('nav-pview:keys')
  })
})

// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { stubResizeObserver } from '@/test/jsdom'
import type { ProfileInfo } from '@/types/hermes'

// Keep store/profile's side-effecting imports inert — same seam as
// store/profile.test.ts / profile-tag.test.tsx.
vi.mock('@/store/gateway', () => ({
  $gateway: atom<unknown>(null),
  activeGateway: vi.fn(() => null),
  ensureGatewayForAgent: vi.fn(async () => undefined),
  ensureGatewayForProfile: vi.fn(async () => undefined),
  openGatewayForProfile: vi.fn(async () => undefined),
  requestGatewayForAgent: vi.fn(async (_connectionId: null | string, _profile: string, method: string) =>
    method === 'vault.sources' ? { sources: [] } : { items: [] }
  )
}))
vi.mock('@/hermes', () => ({
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  setApiRequestProfile: vi.fn()
}))
vi.mock('@/lib/query-client', () => ({ invalidateProfileScopedQueries: vi.fn() }))
vi.mock('@/store/starmap', () => ({ resetStarmapGraph: vi.fn() }))

const { getProfiles } = await import('@/hermes')
const { $activeGatewayProfile, $profiles } = await import('@/store/profile')
const { $gatewayState } = await import('@/store/session')
const { $settingsScopeOverride } = await import('@/store/settings-scope')
const { ActiveProfileNote, SettingsProfileScope } = await import('./profile-scope')
const { VaultSettings } = await import('./vault-settings')

stubResizeObserver()

const profile = (name: string, isDefault = false, extra: Partial<ProfileInfo> = {}): ProfileInfo =>
  ({ has_env: false, is_default: isDefault, model: null, name, ...extra }) as ProfileInfo

beforeEach(() => {
  $activeGatewayProfile.set('default')
  $settingsScopeOverride.set(null)
  $profiles.set([])
})

afterEach(() => {
  cleanup()
  vi.mocked(getProfiles).mockReset().mockResolvedValue({ profiles: [] })
})

describe('SettingsProfileScope', () => {
  it('renders nothing with fewer than two profiles', () => {
    $profiles.set([profile('default', true)])

    const { container } = render(<SettingsProfileScope />)
    expect(container.textContent).toBe('')
  })

  it('selecting another profile sets the shared override; re-selecting the active clears it', () => {
    $profiles.set([profile('default', true), profile('coder')])

    render(<SettingsProfileScope />)

    fireEvent.click(screen.getByRole('button', { name: 'coder' }))
    expect($settingsScopeOverride.get()).toBe('coder')

    fireEvent.click(screen.getByRole('button', { name: 'default' }))
    expect($settingsScopeOverride.get()).toBeNull()
  })

  // After opening a Bot Mode chat, the ACTIVE profile is the bot's, so the
  // settings pages silently edit the bot's config with no override. The target
  // must be stated (accented) whenever it isn't the default profile, override or not.
  it('states the edit target when the active profile is a non-default bot (no override)', () => {
    $activeGatewayProfile.set('scout')
    $profiles.set([profile('default', true), profile('scout')])

    const { container } = render(<SettingsProfileScope />)

    expect($settingsScopeOverride.get()).toBeNull()
    expect(container.textContent).toContain('scout')
    // The note is present and flagged loud (data-scope-loud marks the accented variant).
    const note = container.querySelector('[role="status"]')
    expect(note).toBeTruthy()
    expect(note?.getAttribute('data-scope-loud')).toBe('true')
  })

  it('shows no note when following the active DEFAULT profile', () => {
    $activeGatewayProfile.set('default')
    $profiles.set([profile('default', true), profile('coder')])

    const { container } = render(<SettingsProfileScope />)

    expect(container.querySelector('[role="status"]')).toBeNull()
  })

  it('keeps the quiet note style for an explicit override onto the default profile', () => {
    $activeGatewayProfile.set('scout')
    $profiles.set([profile('default', true), profile('scout')])

    render(<SettingsProfileScope />)

    fireEvent.click(screen.getByRole('button', { name: 'default' }))
    expect($settingsScopeOverride.get()).toBe('default')

    const note = screen.getByRole('status')
    expect(note).toBeTruthy()
    expect(note.hasAttribute('data-scope-loud')).toBe(false)
  })

  it('labels chips with the bot title, else the display name, else the slug', () => {
    $profiles.set([
      profile('default', true, { bot_title: 'JordyV', display_name: 'JordieF' }),
      profile('default-2', false, { display_name: 'Copy' }),
      profile('weather-man')
    ])

    render(<SettingsProfileScope />)

    // Bot Mode title wins over display_name and the slug — same identity the
    // Bots roster shows.
    expect(screen.getByRole('button', { name: 'JordyV' })).toBeTruthy()
    // display_name (profile.yaml) when no Bot Mode title exists.
    expect(screen.getByRole('button', { name: 'Copy' })).toBeTruthy()
    // Canonical slug when neither is set.
    expect(screen.getByRole('button', { name: 'weather-man' })).toBeTruthy()
  })

  it('keeps selection keyed on the canonical name while showing the presentation label', () => {
    $profiles.set([profile('default', true), profile('coder', false, { bot_title: 'JordyV' })])

    render(<SettingsProfileScope />)

    fireEvent.click(screen.getByRole('button', { name: 'JordyV' }))
    // The label changed, the identity did not: the override stores the slug.
    expect($settingsScopeOverride.get()).toBe('coder')
    // The "applies to" note names the target the way its chip does.
    const note = screen.getByRole('status')
    expect(note.textContent).toContain('JordyV')
    expect(note.textContent).not.toContain('coder')
  })
})

// Local Models sends unscoped requests, so it always edits the ACTIVE profile;
// the note must say which one — and stay silent for single-profile users, like
// the selector.
describe('ActiveProfileNote', () => {
  it('names the active profile (by its chip label) only with two or more profiles', () => {
    $activeGatewayProfile.set('setup')
    $profiles.set([profile('default', true)])
    const { container, rerender } = render(<ActiveProfileNote />)
    expect(container.textContent).toBe('')

    $profiles.set([profile('default', true), profile('setup', false, { display_name: 'Setup box' })])
    rerender(<ActiveProfileNote />)
    expect(screen.getByRole('status').textContent).toContain('Setup box')
    expect($settingsScopeOverride.get()).toBeNull()
  })
})

it('renders the shared profile scope on the vault page only when multiple profiles exist', async () => {
  const previousProfiles = $profiles.get()
  const previousProfile = $activeGatewayProfile.get()
  const previousScope = $settingsScopeOverride.get()
  const previousGatewayState = $gatewayState.get()
  const queryClient = new QueryClient()
  const multipleProfiles = [profile('default', true), profile('credentials-owner')]
  const singleProfile = [profile('default', true)]
  const views: ReturnType<typeof render>[] = []

  const renderVault = () => {
    const view = render(
      <MemoryRouter>
        <QueryClientProvider client={queryClient}>
          <VaultSettings />
        </QueryClientProvider>
      </MemoryRouter>
    )

    views.push(view)

    return view
  }

  try {
    $activeGatewayProfile.set('default')
    $settingsScopeOverride.set(null)
    $gatewayState.set('open')
    $profiles.set(multipleProfiles)
    vi.mocked(getProfiles).mockResolvedValueOnce({ profiles: multipleProfiles })
    const multiple = renderVault()

    expect(screen.queryByText('Applies to')).not.toBeNull()
    await act(async () => {
      await vi.mocked(getProfiles).mock.results[0].value
    })
    await act(async () => multiple.unmount())
    views.splice(views.indexOf(multiple), 1)

    $profiles.set(singleProfile)
    vi.mocked(getProfiles).mockResolvedValueOnce({ profiles: singleProfile })
    const single = renderVault()

    expect(screen.queryByText('Applies to')).toBeNull()
    await act(async () => {
      await vi.mocked(getProfiles).mock.results[1].value
    })
    expect(screen.queryByText('Applies to')).toBeNull()
    await act(async () => single.unmount())
    views.splice(views.indexOf(single), 1)
  } finally {
    await act(async () => {
      views.forEach(view => view.unmount())
      queryClient.clear()
      $profiles.set(previousProfiles)
      $activeGatewayProfile.set(previousProfile)
      $settingsScopeOverride.set(previousScope)
      $gatewayState.set(previousGatewayState)
    })
  }
})

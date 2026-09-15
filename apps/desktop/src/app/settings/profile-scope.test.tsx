// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ProfileInfo } from '@/types/hermes'

// Keep store/profile's side-effecting imports inert — same seam as
// store/profile.test.ts / profile-tag.test.tsx.
vi.mock('@/store/gateway', () => ({
  $gateway: atom<unknown>(null),
  ensureGatewayForAgent: vi.fn(async () => undefined),
  ensureGatewayForProfile: vi.fn(async () => undefined),
  openGatewayForProfile: vi.fn(async () => undefined)
}))
vi.mock('@/hermes', () => ({
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  setApiRequestProfile: vi.fn()
}))
vi.mock('@/lib/query-client', () => ({ invalidateProfileScopedQueries: vi.fn() }))
vi.mock('@/store/starmap', () => ({ resetStarmapGraph: vi.fn() }))

const { $activeGatewayProfile, $profiles } = await import('@/store/profile')
const { $settingsScopeOverride } = await import('@/store/settings-scope')
const { SettingsProfileScope } = await import('./profile-scope')

const profile = (name: string, isDefault = false, extra: Partial<ProfileInfo> = {}): ProfileInfo =>
  ({ has_env: false, is_default: isDefault, model: null, name, ...extra }) as ProfileInfo

beforeEach(() => {
  $activeGatewayProfile.set('default')
  $settingsScopeOverride.set(null)
  $profiles.set([])
})

afterEach(cleanup)

describe('SettingsProfileScope', () => {
  it('renders nothing with fewer than two profiles', () => {
    $profiles.set([profile('default', true)])

    const { container } = render(<SettingsProfileScope />)
    expect(container.textContent).toBe('')
  })

  it('shows one chip per profile with the active profile selected by default', () => {
    $profiles.set([profile('default', true), profile('coder')])

    render(<SettingsProfileScope />)

    expect(screen.getByRole('button', { name: 'default' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'coder' })).toBeTruthy()
    // Following the active profile → no override, no "applies to X" note.
    expect($settingsScopeOverride.get()).toBeNull()
  })

  it('selecting another profile sets the shared override; re-selecting the active clears it', () => {
    $profiles.set([profile('default', true), profile('coder')])

    render(<SettingsProfileScope />)

    fireEvent.click(screen.getByRole('button', { name: 'coder' }))
    expect($settingsScopeOverride.get()).toBe('coder')

    fireEvent.click(screen.getByRole('button', { name: 'default' }))
    expect($settingsScopeOverride.get()).toBeNull()
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
  })
})

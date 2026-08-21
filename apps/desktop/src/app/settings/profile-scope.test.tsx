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

const profile = (name: string, isDefault = false): ProfileInfo =>
  ({ has_env: false, is_default: isDefault, model: null, name }) as unknown as ProfileInfo

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

  // #89190/#89162 class: after opening a Bot Mode chat, the ACTIVE profile is
  // the bot's — so with no override the settings pages silently edit the bot's
  // config. The target must be stated (accented) whenever it isn't the default
  // profile, override or not.
  it('states the edit target when the active profile is a non-default bot (no override)', () => {
    $activeGatewayProfile.set('scout')
    $profiles.set([profile('default', true), profile('scout')])

    const { container } = render(<SettingsProfileScope />)

    expect($settingsScopeOverride.get()).toBeNull()
    expect(container.textContent).toContain('scout')
    // The note paragraph is present and accented (font-medium marks the loud variant).
    const note = container.querySelector('p')
    expect(note).toBeTruthy()
    expect(note?.className).toContain('font-medium')
  })

  it('shows no note when following the active DEFAULT profile', () => {
    $activeGatewayProfile.set('default')
    $profiles.set([profile('default', true), profile('coder')])

    const { container } = render(<SettingsProfileScope />)

    expect(container.querySelector('p')).toBeNull()
  })

  it('keeps the quiet note style for an explicit override onto the default profile', () => {
    $activeGatewayProfile.set('scout')
    $profiles.set([profile('default', true), profile('scout')])

    render(<SettingsProfileScope />)

    fireEvent.click(screen.getByRole('button', { name: 'default' }))
    expect($settingsScopeOverride.get()).toBe('default')

    const note = document.querySelector('p')
    expect(note).toBeTruthy()
    expect(note?.className).not.toContain('font-medium')
  })
})

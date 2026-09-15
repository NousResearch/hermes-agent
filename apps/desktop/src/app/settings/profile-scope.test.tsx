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

const profile = (name: string, over: Partial<ProfileInfo> = {}): ProfileInfo =>
  ({ has_env: false, is_default: false, model: null, name, ...over }) as unknown as ProfileInfo

beforeEach(() => {
  $activeGatewayProfile.set('default')
  $settingsScopeOverride.set(null)
  $profiles.set([])
})

afterEach(cleanup)

describe('SettingsProfileScope', () => {
  it('renders nothing with fewer than two profiles', () => {
    $profiles.set([profile('default', { is_default: true })])

    const { container } = render(<SettingsProfileScope />)
    expect(container.textContent).toBe('')
  })

  it('shows Bot title, then display name, while selection keeps the canonical profile id', () => {
    $profiles.set([
      profile('default', {
        display_name: 'JordieF',
        is_default: true,
        ui_meta: { 'hermes-bots': { title: 'JordyV' } }
      }),
      profile('default-2', { display_name: 'JordyV (copy)' }),
      profile('weather-man')
    ])

    render(<SettingsProfileScope />)

    expect(screen.getByRole('button', { name: 'JordyV' })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'JordyV (copy)' }))
    expect($settingsScopeOverride.get()).toBe('default-2')
    expect(screen.getByRole('button', { name: 'weather-man' })).toBeTruthy()
  })

  it('selecting another profile sets the shared override; re-selecting the active clears it', () => {
    $profiles.set([profile('default', { is_default: true }), profile('coder')])

    render(<SettingsProfileScope />)

    fireEvent.click(screen.getByRole('button', { name: 'coder' }))
    expect($settingsScopeOverride.get()).toBe('coder')

    fireEvent.click(screen.getByRole('button', { name: 'default' }))
    expect($settingsScopeOverride.get()).toBeNull()
  })
})

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $activeGatewayProfile, $profileColors, $profileOrder, $profiles, setShowAllProfiles } from '@/store/profile'
import type { ProfileInfo } from '@/types/hermes'

import { ProfileRail } from './profile-switcher'

const getProfiles = vi.hoisted(() => vi.fn<() => Promise<{ profiles: ProfileInfo[] }>>(async () => ({ profiles: [] })))

vi.mock('@/hermes', () => ({
  createProfile: vi.fn(async () => undefined),
  deleteProfile: vi.fn(async () => undefined),
  getProfiles,
  renameProfile: vi.fn(async () => undefined),
  updateProfileSoul: vi.fn(async () => undefined),
  setApiRequestProfile: vi.fn()
}))

vi.mock('@/store/gateway', () => ({
  $gateway: { get: () => null, subscribe: vi.fn(() => () => undefined) },
  ensureGatewayForProfile: vi.fn(async () => undefined)
}))

vi.mock('@/lib/query-client', () => ({ queryClient: { invalidateQueries: vi.fn() } }))

function profile(name: string, overrides: Partial<ProfileInfo> = {}): ProfileInfo {
  return {
    has_env: false,
    is_default: false,
    model: null,
    name,
    path: `/profiles/${name}`,
    provider: null,
    skill_count: 0,
    ...overrides
  }
}

function profilesFixture(): ProfileInfo[] {
  return [profile('default', { is_default: true }), profile('work'), profile('personal')]
}

function renderRail() {
  return render(
    <I18nProvider configClient={null}>
      <MemoryRouter>
        <ProfileRail />
      </MemoryRouter>
    </I18nProvider>
  )
}

beforeEach(() => {
  const profiles = profilesFixture()

  getProfiles.mockResolvedValue({ profiles })
  $profiles.set(profiles)
  $activeGatewayProfile.set('default')
  $profileOrder.set([])
  $profileColors.set({})
  setShowAllProfiles(false)
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  $profiles.set([])
  $profileOrder.set([])
  $profileColors.set({})
  setShowAllProfiles(false)
  $activeGatewayProfile.set('default')
})

describe('ProfileRail', () => {
  it('shows the default profile as a visible option separate from all profiles', () => {
    renderRail()

    const allProfiles = screen.getByRole('button', { name: 'All profiles' })
    const defaultProfile = screen.getByRole('button', { name: 'Switch to default' })

    expect(allProfiles).not.toBe(defaultProfile)
    expect(allProfiles.getAttribute('aria-pressed')).toBe('false')
    expect(defaultProfile.getAttribute('aria-pressed')).toBe('true')
    expect(defaultProfile.textContent).toContain('default')
  })

  it('keeps all profiles and concrete profile selection mutually exclusive', async () => {
    renderRail()

    fireEvent.click(screen.getByRole('button', { name: 'All profiles' }))

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'All profiles' }).getAttribute('aria-pressed')).toBe('true')
      expect(screen.getByRole('button', { name: 'Switch to default' }).getAttribute('aria-pressed')).toBe('false')
    })

    fireEvent.click(screen.getByRole('button', { name: 'Switch to default' }))

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'All profiles' }).getAttribute('aria-pressed')).toBe('false')
      expect(screen.getByRole('button', { name: 'Switch to default' }).getAttribute('aria-pressed')).toBe('true')
    })
  })
})

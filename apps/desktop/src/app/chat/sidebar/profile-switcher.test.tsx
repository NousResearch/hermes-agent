import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $profileOrder, $profiles, $selectedProfileScope, $showAllProfiles } from '@/store/profile'
import {
  $attentionSessionIds,
  $cronSessions,
  $messagingSessions,
  $replyReadySessionIds,
  $replyReadySessionProfiles,
  $sessions,
  $workingSessionIds
} from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { ProfileRail } from './profile-switcher'

const getProfilesMock = vi.hoisted(() => vi.fn())

vi.mock('@/hermes', () => ({
  getProfiles: getProfilesMock,
  setApiRequestProfile: vi.fn()
}))

vi.mock('@/store/gateway', () => ({
  $gateway: { get: () => null, set: vi.fn(), subscribe: vi.fn(() => () => {}) },
  ensureGatewayForProfile: vi.fn(async () => undefined)
}))

vi.mock('@/lib/query-client', () => ({ queryClient: { invalidateQueries: vi.fn() } }))

function sessionInfo(overrides: Partial<SessionInfo> = {}): SessionInfo {
  return {
    ended_at: null,
    id: 'session-1',
    input_tokens: 0,
    is_active: true,
    last_active: 1000,
    message_count: 1,
    model: null,
    output_tokens: 0,
    preview: null,
    source: null,
    started_at: 1000,
    title: 'Session',
    tool_call_count: 0,
    ...overrides
  }
}

function profileInfo(name: string, isDefault = false) {
  return {
    has_env: true,
    is_default: isDefault,
    model: null,
    name,
    path: isDefault ? '/home/endlessblink/.hermes' : `/home/endlessblink/.hermes/profiles/${name}`,
    provider: null,
    skill_count: 0
  }
}

function mockRailMetrics(el: HTMLElement, { clientWidth, scrollLeft, scrollWidth }: {
  clientWidth: number
  scrollLeft: number
  scrollWidth: number
}) {
  let currentScrollLeft = scrollLeft

  Object.defineProperties(el, {
    clientWidth: { configurable: true, get: () => clientWidth },
    scrollLeft: {
      configurable: true,
      get: () => currentScrollLeft,
      set: value => {
        currentScrollLeft = Number(value)
      }
    },
    scrollWidth: { configurable: true, get: () => scrollWidth }
  })
}

describe('ProfileRail attention badges', () => {
  beforeEach(() => {
    getProfilesMock.mockImplementation(async () => ({ profiles: $profiles.get() }))
    $profiles.set([
      {
        has_env: true,
        is_default: true,
        model: null,
        name: 'default',
        path: '/home/endlessblink/.hermes',
        provider: null,
        skill_count: 0
      },
      {
        has_env: true,
        is_default: false,
        model: null,
        name: 'film-maker',
        path: '/home/endlessblink/.hermes/profiles/film-maker',
        provider: null,
        skill_count: 0
      }
    ])
    $profileOrder.set([])
    $selectedProfileScope.set('default')
    $showAllProfiles.set(false)
    $sessions.set([
      sessionInfo({ id: 's1', profile: 'film-maker', title: 'Waiting one' }),
      sessionInfo({ id: 's2', profile: 'film-maker', title: 'Waiting two' }),
      sessionInfo({ id: 's3', profile: 'default', title: 'Idle default' })
    ])
    $cronSessions.set([])
    $messagingSessions.set([])
    $workingSessionIds.set([])
    $attentionSessionIds.set([])
    $replyReadySessionIds.set(['s1', 's2'])
    $replyReadySessionProfiles.set({ s1: 'film-maker', s2: 'film-maker' })
    Object.assign(window, { hermesDesktop: { api: vi.fn(async () => ({ current: 'default' })) } })
  })

  afterEach(() => {
    cleanup()
    getProfilesMock.mockReset()
    Reflect.deleteProperty(window, 'hermesDesktop')
    $profiles.set([])
    $profileOrder.set([])
    $selectedProfileScope.set('default')
    $showAllProfiles.set(false)
    $sessions.set([])
    $cronSessions.set([])
    $messagingSessions.set([])
    $workingSessionIds.set([])
    $attentionSessionIds.set([])
    $replyReadySessionIds.set([])
    $replyReadySessionProfiles.set({})
  })

  it('shows a waiting-reply count on the owning profile square', () => {
    render(
      <MemoryRouter>
        <ProfileRail />
      </MemoryRouter>
    )

    const profileButton = screen.getByRole('button', { name: /film-maker, 2 waiting for your reply/i })

    expect(profileButton.textContent).toContain('2')
  })

  it('keeps waiting-reply counts in the profile rail layout flow', () => {
    render(
      <MemoryRouter>
        <ProfileRail />
      </MemoryRouter>
    )

    const profileButton = screen.getByRole('button', { name: /film-maker, 2 waiting for your reply/i })
    const countChip = profileButton.querySelector('[data-profile-count]')

    expect(profileButton.className).toContain('h-5')
    expect(profileButton.className).toContain('min-w-7')
    expect(countChip?.className).toContain('font-mono')
    expect(countChip?.className).not.toContain('absolute')
  })

  it('does not refresh profiles from the rail mount path when the cache is warm', () => {
    const api = vi.fn(async () => ({ current: 'default' }))
    Object.assign(window, { hermesDesktop: { api } })

    render(
      <MemoryRouter>
        <ProfileRail />
      </MemoryRouter>
    )

    expect(api).not.toHaveBeenCalled()
    expect(getProfilesMock).not.toHaveBeenCalled()
  })

  it('refreshes profiles when the rail mounts with an empty cache', async () => {
    const api = vi.fn(async () => ({ current: 'default' }))
    Object.assign(window, { hermesDesktop: { api } })
    $profiles.set([])
    getProfilesMock.mockResolvedValueOnce({
      profiles: [profileInfo('default', true), profileInfo('bina-meatzevet')]
    })

    render(
      <MemoryRouter>
        <ProfileRail />
      </MemoryRouter>
    )

    expect(await screen.findByRole('button', { name: 'bina-meatzevet' })).toBeDefined()
    expect(api).toHaveBeenCalledWith({ path: '/api/profiles/active' })
    expect(getProfilesMock).toHaveBeenCalledTimes(1)
  })

  it('exposes overflowed profiles with explicit rail scroll controls', async () => {
    $profiles.set([
      profileInfo('default', true),
      profileInfo('bina-meatzevet'),
      profileInfo('content-creator'),
      profileInfo('film-maker'),
      profileInfo('finding-jobs-and-projects'),
      profileInfo('hermes-dev'),
      profileInfo('office-work'),
      profileInfo('research')
    ])

    render(
      <MemoryRouter>
        <ProfileRail />
      </MemoryRouter>
    )

    const rail = screen.getByLabelText('Profile list')
    mockRailMetrics(rail, { clientWidth: 52, scrollLeft: 0, scrollWidth: 220 })
    fireEvent.scroll(rail)

    const next = await screen.findByRole('button', { name: /Scroll profiles right, 2 waiting for your reply/i })

    expect(screen.getByRole('button', { name: 'office-work' })).toBeDefined()
    expect(next.className).toContain('h-5')
    expect(next.className).toContain('min-w-7')
    expect(next.textContent).toContain('2')
    expect(screen.getByRole('button', { name: 'Scroll profiles left' }).hasAttribute('disabled')).toBe(true)

    fireEvent.click(next)

    await waitFor(() => {
      expect(rail.scrollLeft).toBeGreaterThan(0)
    })

    fireEvent.click(screen.getByRole('button', { name: 'office-work' }))

    expect($selectedProfileScope.get()).toBe('office-work')
  })
})

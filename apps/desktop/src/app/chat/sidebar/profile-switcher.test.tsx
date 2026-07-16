import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $activeGatewayProfile, $profileColors, $profileOrder, $profiles, $showAllProfiles } from '@/store/profile'
import { $attentionSessionIds, $sessions, $unreadFinishedSessionIds, $workingSessionIds } from '@/store/session'
import { $subagentsBySession, upsertSubagent } from '@/store/subagents'
import type { ProfileInfo, SessionInfo } from '@/types/hermes'

import { ProfileRail } from './profile-switcher'

const profile = (name: string, isDefault = false): ProfileInfo => ({
  has_env: false,
  is_default: isDefault,
  model: null,
  name,
  path: `/tmp/hermes/${name}`,
  provider: null,
  skill_count: 0
})

const session = (id: string, owner: string): SessionInfo =>
  ({
    id,
    profile: owner
  }) as SessionInfo

const defaultProfiles = [profile('default', true), profile('claire'), profile('wallace')]
let profiles = defaultProfiles
let activeProfile = 'default'

function renderRail() {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <MemoryRouter>
        <ProfileRail />
      </MemoryRouter>
    </I18nProvider>
  )
}

describe('ProfileRail activity indicators', () => {
  beforeEach(() => {
    profiles = defaultProfiles
    activeProfile = 'default'
    $profiles.set(profiles)
    $activeGatewayProfile.set('default')
    $showAllProfiles.set(false)
    $profileOrder.set([])
    $profileColors.set({})
    $sessions.set([
      session('default-run', 'default'),
      session('claire-run', 'claire'),
      session('wallace-done', 'wallace')
    ])
    $workingSessionIds.set(['default-run', 'claire-run'])
    $attentionSessionIds.set([])
    $unreadFinishedSessionIds.set(['wallace-done'])
    $subagentsBySession.set({})

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        api: vi.fn(async ({ path }: { path: string }) =>
          path === '/api/profiles/active' ? { current: activeProfile } : { profiles }
        )
      }
    })
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    $sessions.set([])
    $workingSessionIds.set([])
    $attentionSessionIds.set([])
    $unreadFinishedSessionIds.set([])
    $subagentsBySession.set({})
  })

  it('brightens and animates profile controls for running and unread sessions', async () => {
    renderRail()

    const defaultButton = screen.getByRole('button', { name: 'Show all profiles · Session running' })
    const claireButton = screen.getByRole('button', { name: 'claire · Session running' })
    const wallaceButton = screen.getByRole('button', { name: 'wallace · New result, not viewed' })

    expect(defaultButton.getAttribute('data-profile-activity')).toBe('working')
    expect(claireButton.getAttribute('data-profile-activity')).toBe('working')
    expect(wallaceButton.getAttribute('data-profile-activity')).toBe('unread')
    expect(claireButton.querySelector('[data-profile-activity-border="working"]')).toBeTruthy()
    expect(wallaceButton.querySelector('[data-profile-activity-border="unread"]')).toBeTruthy()
    expect(wallaceButton.querySelector('[data-profile-activity-pip="unread"]')).toBeTruthy()

    await waitFor(() => expect(vi.mocked(window.hermesDesktop.api)).toHaveBeenCalled())
  })

  it('promotes a blocking prompt above running and unread activity', () => {
    renderRail()

    act(() => {
      $attentionSessionIds.set(['claire-run'])
      $unreadFinishedSessionIds.set(['claire-run', 'wallace-done'])
    })

    const claireButton = screen.getByRole('button', { name: 'claire · Needs input' })
    expect(claireButton.getAttribute('data-profile-activity')).toBe('needs-input')
    expect(claireButton.querySelector('[data-profile-activity-border="needs-input"]')).toBeTruthy()
    expect(claireButton.querySelector('[data-profile-activity-pip="needs-input"]')).toBeTruthy()
  })

  it('keeps the parent profile active while an independent review subagent runs', () => {
    $workingSessionIds.set([])
    $unreadFinishedSessionIds.set(['claire-run'])
    upsertSubagent(
      'runtime-claire',
      {
        child_session_id: 'claire-review',
        goal: 'Independent review',
        status: 'running',
        subagent_id: 'review-1'
      },
      true,
      'subagent.start',
      'claire-run'
    )

    renderRail()

    expect(screen.getByRole('button', { name: 'claire · Session running' }).getAttribute('data-profile-activity')).toBe(
      'working'
    )
  })

  it('surfaces the strongest hidden profile activity on the condensed trigger', () => {
    profiles = [profile('default', true), ...Array.from({ length: 14 }, (_, index) => profile(`p${index + 1}`))]
    activeProfile = 'p1'
    $profiles.set(profiles)
    $activeGatewayProfile.set(activeProfile)
    $sessions.set([session('p12-waiting', 'p12'), session('p13-finished', 'p13')])
    $workingSessionIds.set(['p12-waiting'])
    $attentionSessionIds.set(['p12-waiting'])
    $unreadFinishedSessionIds.set(['p13-finished'])

    renderRail()

    const trigger = screen.getByRole('combobox', { name: 'p1' })
    const description = globalThis.document.getElementById(trigger.getAttribute('aria-describedby') ?? '')

    expect(description?.textContent).toBe('p12 · Needs input')
    expect(trigger.getAttribute('data-profile-activity')).toBe('needs-input')
    expect(trigger.querySelector('[data-profile-activity-pip="needs-input"]')).toBeTruthy()
  })
})

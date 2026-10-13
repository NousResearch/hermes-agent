import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $activeGatewayProfile, $newChatProfile, $profiles } from '@/store/profile'
import { $projectTree } from '@/store/projects'
import { $cronSessions, $sessions, $unlistedSessionOwnerRows } from '@/store/session'
import type { ProfileInfo, SessionInfo } from '@/types/hermes'

import { SessionTabProfileTag } from './session-tab-profile-tag'

function renderLead(storedSessionId: null | string) {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <SessionTabProfileTag storedSessionId={storedSessionId} />
    </I18nProvider>
  )
}

const row = (overrides: Partial<SessionInfo>): SessionInfo => ({ id: 's1', ...overrides }) as SessionInfo

const profile = (name: string, isDefault = false): ProfileInfo =>
  ({ has_env: false, is_default: isDefault, model: null, name }) as ProfileInfo

const lead = (name: string) => screen.getByRole('img', { name: `Profile: ${name}` })

function reset() {
  $sessions.set([])
  $cronSessions.set([])
  $unlistedSessionOwnerRows.set([])
  $projectTree.set([])
  $newChatProfile.set(null)
  $activeGatewayProfile.set('default')
  // Multi-profile install by default: the lead only renders once a second profile exists.
  $profiles.set([profile('default', true), profile('omar')])
}

beforeEach(reset)

afterEach(() => {
  cleanup()
  reset()
})

describe('SessionTabProfileTag — the identity at the start of a session tab', () => {
  it('shows a stored session’s owning profile as glyph + name under the "Profile: …" tip', () => {
    $sessions.set([row({ id: 'stored-1', profile: 'code-reviewer' })])
    renderLead('stored-1')

    expect(screen.getByText('code-reviewer')).toBeTruthy()
    expect(lead('code-reviewer')).toBeTruthy()
  })

  it('the visible name is aria-hidden, so the owner is announced once (via the glyph label)', () => {
    $sessions.set([row({ id: 'stored-1', profile: 'ops' })])
    renderLead('stored-1')

    expect(screen.getByText('ops').getAttribute('aria-hidden')).toBe('true')
  })

  it('renders nothing on a single-profile install (same rule as the chat header)', () => {
    $profiles.set([profile('default', true)])
    $sessions.set([row({ id: 'stored-1', profile: 'default' })])
    const { container } = renderLead('stored-1')

    expect(container.textContent).toBe('')
  })

  it('a draft shows the profile it would be created under ($newChatProfile)', () => {
    $newChatProfile.set('research')
    renderLead(null)

    expect(lead('research')).toBeTruthy()
  })

  it('a draft with no new-chat pick inherits the live gateway profile, like the create path', () => {
    $activeGatewayProfile.set('omar')
    renderLead(null)

    expect(lead('omar')).toBeTruthy()
  })

  it('an unlisted ⌘T tab reads its owner stub, not the new-chat pick', () => {
    $unlistedSessionOwnerRows.set([row({ id: 'tile-1', profile: 'omar' })])
    renderLead('tile-1')
    expect(lead('omar')).toBeTruthy()

    act(() => $newChatProfile.set('research'))
    expect(lead('omar')).toBeTruthy()
  })

  it('resolves owners from the cron slice and the project tree', () => {
    $cronSessions.set([row({ id: 'cron-1', profile: 'ops' })])
    $projectTree.set([
      { previewSessions: [], repos: [{ groups: [{ sessions: [row({ id: 'proj-1', profile: 'proj' })] }] }] }
    ] as never)

    renderLead('cron-1')
    expect(lead('ops')).toBeTruthy()
    cleanup()

    renderLead('proj-1')
    expect(lead('proj')).toBeTruthy()
  })

  it('a stored row outranks draft intent', () => {
    $sessions.set([row({ id: 'x', profile: 'ops' })])
    $newChatProfile.set('research')
    renderLead('x')

    expect(lead('ops')).toBeTruthy()
  })

  it('is self-subscribing: identity follows live store changes without the strip re-registering', () => {
    renderLead('stored-2')

    // No row yet → the profile its first turn would be created under.
    expect(lead('default')).toBeTruthy()

    // First turn mints the stored row: the lead adopts its owning profile.
    act(() => $sessions.set([row({ id: 'stored-2', profile: 'ops' })]))
    expect(lead('ops')).toBeTruthy()

    // …and keeps tracking the row if the profile on it changes.
    act(() => $sessions.set([row({ id: 'stored-2', profile: 'review-b' })]))
    expect(lead('review-b')).toBeTruthy()
  })
})

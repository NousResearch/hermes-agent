import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $newChatProfile } from '@/store/profile'
import { $sessions } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { SessionTabProfileTag } from './session-tab-profile-tag'

function renderLead(storedSessionId: null | string) {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <SessionTabProfileTag storedSessionId={storedSessionId} />
    </I18nProvider>
  )
}

const row = (overrides: Partial<SessionInfo>): SessionInfo => ({ id: 's1', ...overrides }) as SessionInfo

beforeEach(() => {
  $sessions.set([])
  $newChatProfile.set(null)
})

afterEach(() => {
  cleanup()
  $sessions.set([])
  $newChatProfile.set(null)
})

describe('SessionTabProfileTag — the identity at the start of a session tab', () => {
  it('shows a stored session’s owning profile as glyph + name under the "Profile: …" tip', () => {
    $sessions.set([row({ id: 'stored-1', profile: 'code-reviewer' })])
    renderLead('stored-1')

    expect(screen.getByText('code-reviewer')).toBeTruthy()
    expect(screen.getByRole('img', { name: 'Profile: code-reviewer' })).toBeTruthy()
  })

  it('falls back to the profile the draft would be created under when no row exists', () => {
    $newChatProfile.set('research')
    renderLead(null)

    expect(screen.getByText('research')).toBeTruthy()
    expect(screen.getByRole('img', { name: 'Profile: research' })).toBeTruthy()
  })

  it('reads a profile-less draft as the literal name "default"', () => {
    renderLead('draft-with-no-row')

    expect(screen.getByText('default')).toBeTruthy()
    expect(screen.getByRole('img', { name: 'Profile: default' })).toBeTruthy()
  })

  it('is self-subscribing: identity follows live store changes without the strip re-registering', () => {
    renderLead('stored-2')

    // No row yet → the draft's profile (default), not a frozen capture.
    expect(screen.getByText('default')).toBeTruthy()

    // First turn mints the stored row: the lead adopts its owning profile.
    act(() => $sessions.set([row({ id: 'stored-2', profile: 'ops' })]))
    expect(screen.getByText('ops')).toBeTruthy()

    // …and keeps tracking the row if the profile on it changes.
    act(() => $sessions.set([row({ id: 'stored-2', profile: 'review-b' })]))
    expect(screen.getByText('review-b')).toBeTruthy()
  })
})

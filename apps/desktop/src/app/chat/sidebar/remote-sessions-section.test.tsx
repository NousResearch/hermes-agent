import { cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $sessionPresence, $sessions } from '@/store/session'
import type { SessionPresenceRecord } from '@/types/hermes'

import { SidebarRemoteSessionsSection } from './remote-sessions-section'

const ENDPOINT = 'ws://192.168.1.20:8664/api/ws'

function presence(over: Partial<SessionPresenceRecord> & { session_id: string }): SessionPresenceRecord {
  return { endpoint: ENDPOINT, host: 'ko-win11', title: 'Remote work', ...over }
}

function renderSection(onCreateOnDevice = vi.fn(), onResumeSession = vi.fn()) {
  return render(
    <SidebarRemoteSessionsSection
      label="Live on other devices"
      newSessionLabel={target => `New session in ${target}`}
      onCreateOnDevice={onCreateOnDevice}
      onResumeSession={onResumeSession}
      onToggle={vi.fn()}
      open
    />
  )
}

describe('SidebarRemoteSessionsSection', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-06-11T00:00:00Z'))
    $sessions.set([])
    $sessionPresence.set([])
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    $sessions.set([])
    $sessionPresence.set([])
  })

  it('renders nothing without remote sessions (no presence sync / single device)', () => {
    const { container } = renderSection()

    expect(container.firstChild).toBeNull()
  })

  it('renders a "new session" row per reachable device and fires onCreateOnDevice', () => {
    $sessionPresence.set([presence({ session_id: 'r1' })])
    const onCreate = vi.fn()

    const { getByText } = renderSection(onCreate)
    fireEvent.click(getByText('New session in ko-win11'))

    expect(onCreate).toHaveBeenCalledWith(ENDPOINT)
  })

  it('resumes an existing remote session when its row is clicked', () => {
    $sessionPresence.set([presence({ session_id: 'r1', title: 'Remote work' })])
    const onResume = vi.fn()

    const { getByText } = renderSection(vi.fn(), onResume)
    fireEvent.click(getByText('Remote work'))

    expect(onResume).toHaveBeenCalledWith('r1')
  })

  it('renders existing remote sessions as one-line rows with a right-aligned next-action label', () => {
    $sessionPresence.set([presence({ session_id: 'r1', status: 'idle', title: 'Remote work', updated_at: 1000 })])

    const { container } = renderSection()
    const row = container.querySelector('[data-remote-session-row]') as HTMLElement

    expect(row.textContent).toContain('Remote work')
    expect(row.textContent).toContain('Next')
    expect(row.textContent).not.toContain('ko-win11')
    expect(row.querySelector('[data-remote-session-status]')?.textContent).toBe('Next')
  })

  it('uses timestamps for remote sessions waiting on the user', () => {
    $sessionPresence.set([
      presence({
        session_id: 'r1',
        status: 'waiting',
        title: 'Clarify work',
        updated_at: Date.now() / 1000 - 120
      })
    ])

    const { container } = renderSection()

    expect(container.querySelector('[data-remote-session-status]')?.textContent).toBe('2m')
  })

  it('hides remote row metadata while the session is actively working', () => {
    $sessionPresence.set([presence({ session_id: 'r1', status: 'working', title: 'Live work' })])

    const { container } = renderSection()

    expect(container.querySelector('[data-remote-session-status]')).toBeNull()
  })

  it('falls back to the endpoint host when a peer has no name', () => {
    $sessionPresence.set([presence({ session_id: 'r1', host: undefined })])

    const { getByText } = renderSection()

    // deviceLabel derives "host:port" from the ws endpoint when host is empty.
    expect(getByText('New session in 192.168.1.20:8664')).toBeTruthy()
  })
})

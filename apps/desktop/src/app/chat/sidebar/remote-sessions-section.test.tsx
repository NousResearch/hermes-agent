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
    $sessions.set([])
    $sessionPresence.set([])
  })

  afterEach(() => {
    cleanup()
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

  it('falls back to the endpoint host when a peer has no name', () => {
    $sessionPresence.set([presence({ session_id: 'r1', host: undefined })])

    const { getByText } = renderSection()

    // deviceLabel derives "host:port" from the ws endpoint when host is empty.
    expect(getByText('New session in 192.168.1.20:8664')).toBeTruthy()
  })
})

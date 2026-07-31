import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { StrictMode } from 'react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { queryClient } from '@/lib/query-client'
import { $activeGatewayProfile } from '@/store/profile'
import type { HermesConfigRecord } from '@/types/hermes'

const getHermesConfigRecord = vi.fn()
const saveHermesConfig = vi.fn()
const listAllProfileSessions = vi.fn()

vi.mock('@/hermes', () => ({
  getHermesConfigRecord: () => getHermesConfigRecord(),
  saveHermesConfig: (config: unknown) => saveHermesConfig(config),
  listAllProfileSessions: (limit: number, offset: number, archived: string) =>
    listAllProfileSessions(limit, offset, archived),
  deleteSession: vi.fn(),
  setSessionArchived: vi.fn(),
  // Pulled in via useOnProfileSwitch → @/store/profile.
  getProfiles: async () => ({ profiles: [] }),
  setApiRequestProfile: () => {},
  STARTUP_REQUEST_TIMEOUT_MS: 1000
}))

const profileRecord = (cwd: string, sessions: Record<string, unknown>): HermesConfigRecord => ({
  terminal: { cwd },
  sessions
})

beforeEach(() => {
  // AutoArchiveSetting only fetches config when the Electron bridge exists.
  ;(window as { hermesDesktop?: unknown }).hermesDesktop = {}

  getHermesConfigRecord.mockImplementation(async () => profileRecord('/profile-a', { auto_archive: false }))
  saveHermesConfig.mockResolvedValue({ ok: true })
  listAllProfileSessions.mockResolvedValue({ sessions: [] })
})

afterEach(() => {
  cleanup()
  queryClient.clear()
  $activeGatewayProfile.set('default')
  vi.clearAllMocks()
  delete (window as { hermesDesktop?: unknown }).hermesDesktop
})

async function renderSessionsSettings() {
  const { SessionsSettings } = await import('./sessions-settings')

  return render(
    <StrictMode>
      <MemoryRouter>
        <SessionsSettings />
      </MemoryRouter>
    </StrictMode>
  )
}

describe('AutoArchiveSetting profile switches', () => {
  it('reseeds profile B’s record after a switch and never persists profile A’s copy', async () => {
    await renderSessionsSettings()

    const toggle = await screen.findByRole('switch', { name: 'Auto-archive stale chats' })
    expect(toggle.getAttribute('aria-checked')).toBe('false')

    const callsBeforeSwitch = getHermesConfigRecord.mock.calls.length

    // Switch to profile B, whose record differs in both the toggle state and
    // an unrelated key (terminal.cwd) that the whole-record PUT would carry.
    getHermesConfigRecord.mockImplementation(async () =>
      profileRecord('/profile-b', { auto_archive: true, auto_archive_days: 9 })
    )

    act(() => {
      $activeGatewayProfile.set('coder')
    })

    // The control drops profile A's copy and refetches for B…
    await waitFor(() => expect(getHermesConfigRecord.mock.calls.length).toBeGreaterThan(callsBeforeSwitch))

    // …and reseeds every piece of its state from B's record.
    const reseeded = await screen.findByRole('switch', { name: 'Auto-archive stale chats' })
    await waitFor(() => expect(reseeded.getAttribute('aria-checked')).toBe('true'))
    expect(screen.getByDisplayValue('9')).toBeTruthy()

    // Toggling now must PUT a record based on profile B's copy — before the
    // fix, the stale profile-A record (cwd /profile-a) was written into B.
    fireEvent.click(reseeded)

    await waitFor(() => expect(saveHermesConfig).toHaveBeenCalledTimes(1))
    expect(saveHermesConfig.mock.calls[0][0]).toEqual(
      profileRecord('/profile-b', { auto_archive: false, auto_archive_days: 9 })
    )
  })
})

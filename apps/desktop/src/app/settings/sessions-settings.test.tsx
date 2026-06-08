import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

const listAllProfileSessions = vi.fn()
const setSessionArchived = vi.fn()
const deleteSession = vi.fn()

vi.mock('@/hermes', () => ({
  listAllProfileSessions: (...args: unknown[]) => listAllProfileSessions(...args),
  setSessionArchived: (...args: unknown[]) => setSessionArchived(...args),
  deleteSession: (...args: unknown[]) => deleteSession(...args)
}))

// Provide a real, writable atom so flipping the active profile re-drives the
// component, plus the normalizer it uses — without pulling in the whole profile
// store (gateway pool, query client, …) and its @/hermes side effects.
vi.mock('@/store/profile', async () => {
  const { atom } = await import('nanostores')

  return {
    $activeGatewayProfile: atom<string>('default'),
    normalizeProfileKey: (name: null | string | undefined) => (name ?? '').trim() || 'default'
  }
})

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

vi.mock('@/lib/haptics', () => ({
  triggerHaptic: vi.fn()
}))

const session = (over: Partial<SessionInfo>): SessionInfo => ({
  archived: true,
  cwd: null,
  ended_at: null,
  id: 'a',
  input_tokens: 0,
  is_active: false,
  last_active: 0,
  message_count: 1,
  model: null,
  output_tokens: 0,
  preview: null,
  source: null,
  started_at: 0,
  title: 'Archived chat',
  tool_call_count: 0,
  ...over
})

async function renderSessionsSettings() {
  const { SessionsSettings } = await import('./sessions-settings')

  return render(
    <MemoryRouter initialEntries={['/settings?tab=sessions']}>
      <SessionsSettings />
    </MemoryRouter>
  )
}

const ARCHIVED_FETCH_LIMIT = 200

beforeEach(() => {
  listAllProfileSessions.mockResolvedValue({ sessions: [], total: 0, profile_totals: {}, limit: 200, offset: 0 })
  setSessionArchived.mockResolvedValue({ ok: true })
})

afterEach(async () => {
  cleanup()
  vi.clearAllMocks()
  const { $activeGatewayProfile } = await import('@/store/profile')
  $activeGatewayProfile.set('default')
})

describe('SessionsSettings archived list', () => {
  it('loads archived chats scoped to the active (named) profile', async () => {
    const { $activeGatewayProfile } = await import('@/store/profile')
    $activeGatewayProfile.set('work')

    await renderSessionsSettings()

    await waitFor(() => expect(listAllProfileSessions).toHaveBeenCalled())
    expect(listAllProfileSessions).toHaveBeenCalledWith(ARCHIVED_FETCH_LIMIT, 0, 'only', 'recent', 'work')
  })

  it('falls back to the default profile when no named profile is active', async () => {
    await renderSessionsSettings()

    await waitFor(() => expect(listAllProfileSessions).toHaveBeenCalled())
    expect(listAllProfileSessions).toHaveBeenCalledWith(ARCHIVED_FETCH_LIMIT, 0, 'only', 'recent', 'default')
  })

  it('unarchives against the row’s owning profile, not the primary', async () => {
    const { $activeGatewayProfile } = await import('@/store/profile')
    $activeGatewayProfile.set('work')
    // The cross-profile list tags rows with their owning profile; the mutation
    // must route there so it does not no-op against the primary state.db.
    listAllProfileSessions.mockResolvedValue({
      sessions: [session({ id: 'w1', profile: 'work', is_default_profile: false })],
      total: 1,
      profile_totals: { work: 1 },
      limit: 200,
      offset: 0
    })

    await renderSessionsSettings()

    const unarchive = await screen.findByRole('button', { name: 'Unarchive' })
    fireEvent.click(unarchive)

    await waitFor(() => expect(setSessionArchived).toHaveBeenCalledWith('w1', false, 'work'))
  })
})

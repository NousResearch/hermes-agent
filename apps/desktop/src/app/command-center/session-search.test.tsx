import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import type { SessionInfo, SessionSearchResult } from '@/hermes'
import { $sessions } from '@/store/session'

import { CommandCenterView } from './index'

// #51694: the Command Center's Sessions search filtered only the sidebar's
// loaded page (`$sessions`, 50 rows, archived excluded) with a substring test
// on title+id — so it never did full-text search, never showed archived
// sessions, and never found anything older than the page. It must call the
// server FTS endpoint (searchSessions) on a debounce and merge the hits behind
// the instant client-side matches, exactly like the sidebar does.

const mocks = vi.hoisted(() => ({
  searchSessions: vi.fn()
}))

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  getActionStatus: vi.fn(() => Promise.resolve({ running: false })),
  getLogs: vi.fn(() => Promise.resolve({ lines: [] })),
  getStatus: vi.fn(() => Promise.resolve({})),
  getUsageAnalytics: vi.fn(() => Promise.resolve({})),
  restartGateway: vi.fn(),
  searchSessions: mocks.searchSessions,
  updateHermes: vi.fn()
}))
vi.mock('@/lib/session-export', () => ({ exportSession: vi.fn() }))
vi.mock('./maintenance', () => ({ MaintenancePanel: () => null }))

afterEach(cleanup)

const SESSION = {
  archived: false,
  ended_at: null,
  id: 'sess-local-1',
  input_tokens: 0,
  is_active: false,
  last_active: 1_756_600_000,
  message_count: 3,
  model: null,
  output_tokens: 0,
  started_at: 1_756_500_000,
  title: 'Precious conversation'
} as unknown as SessionInfo

function serverHit(overrides: Partial<SessionSearchResult> & { session_id: string }): SessionSearchResult {
  return {
    archived: false,
    last_active: 1_756_400_000,
    model: null,
    preview: null,
    role: null,
    snippet: 'a snippet',
    session_started: 1_756_300_000,
    source: null,
    started_at: 1_756_300_000,
    title: null,
    ...overrides
  }
}

function renderCommandCenter() {
  return render(
    <MemoryRouter>
      <CommandCenterView
        initialSection="sessions"
        onClose={() => {}}
        onDeleteSession={vi.fn(() => Promise.resolve())}
        onOpenSession={() => {}}
      />
    </MemoryRouter>
  )
}

async function typeSearch(value: string) {
  fireEvent.change(screen.getByPlaceholderText('Search sessions, views, and actions'), {
    target: { value }
  })
}

describe('Command Center sessions server search (#51694)', () => {
  beforeEach(() => {
    mocks.searchSessions.mockReset()
    $sessions.set([SESSION])
  })

  it('calls the server search endpoint and lists a server-only hit', async () => {
    mocks.searchSessions.mockResolvedValue({
      results: [
        serverHit({
          archived: false,
          session_id: 'server-only-1',
          snippet: '...mentions >>>zzqmarker<<< here...',
          title: 'Frozen tundra notes'
        })
      ]
    })
    renderCommandCenter()

    await typeSearch('zzqmarker')

    await waitFor(() => expect(mocks.searchSessions).toHaveBeenCalledWith('zzqmarker'), { timeout: 3000 })
    expect(await screen.findByText('Frozen tundra notes', {}, { timeout: 3000 })).toBeTruthy()
  })

  it('marks an archived server hit with the archived badge', async () => {
    mocks.searchSessions.mockResolvedValue({
      results: [
        serverHit({
          archived: true,
          session_id: 'server-archived-1',
          snippet: 'buried >>>zzqfrozen<<< conversation',
          title: 'Frozen archive thread'
        })
      ]
    })
    renderCommandCenter()

    await typeSearch('zzqfrozen')

    await waitFor(() => expect(mocks.searchSessions).toHaveBeenCalledWith('zzqfrozen'), { timeout: 3000 })
    // The archived marker is a visible text badge, not a tooltip on a glyph.
    expect(await screen.findByText('Frozen archive thread', {}, { timeout: 3000 })).toBeTruthy()
    expect(await screen.findByText('Archived', {}, { timeout: 3000 })).toBeTruthy()
  })

  it('strips the backend FTS highlight markers from rendered text', async () => {
    mocks.searchSessions.mockResolvedValue({
      results: [
        serverHit({
          session_id: 'server-snippet-1',
          snippet: 'mentions >>>zzqalpha<<< here',
          title: null
        })
      ]
    })
    renderCommandCenter()

    await typeSearch('zzqalpha')

    await waitFor(() => expect(mocks.searchSessions).toHaveBeenCalledWith('zzqalpha'), { timeout: 3000 })
    // title is null → the row falls back to the (marker-stripped) snippet.
    expect(await screen.findByText('mentions zzqalpha here', {}, { timeout: 3000 })).toBeTruthy()
    expect(screen.queryByText(/>>>|<<</)).toBeNull()
  })

  it('does not call the server with no query and still lists loaded sessions', async () => {
    renderCommandCenter()

    expect(await screen.findByText('Precious conversation', {}, { timeout: 3000 })).toBeTruthy()
    expect(mocks.searchSessions).not.toHaveBeenCalled()
  })

  it('renders the loaded local match immediately and before the server hit', async () => {
    // The query matches the loaded session's own title, so the client-side
    // pass can answer without the backend; the server returns a different
    // conversation. Local-first means the loaded row paints while the request
    // is in flight and stays above the server hit once it lands.
    let resolveSearch: ((value: { results: SessionSearchResult[] }) => void) | undefined

    mocks.searchSessions.mockImplementation(
      () =>
        new Promise<{ results: SessionSearchResult[] }>(resolve => {
          resolveSearch = resolve
        })
    )
    renderCommandCenter()

    await typeSearch('Precious')

    // Wait for the debounce to fire and the request to be in flight (the
    // promise below is still unresolved); the loaded local match must already
    // be on screen without waiting for the backend.
    await waitFor(() => expect(mocks.searchSessions).toHaveBeenCalledWith('Precious'), { timeout: 3000 })
    expect(await screen.findByText('Precious conversation', {}, { timeout: 3000 })).toBeTruthy()

    await act(async () => {
      resolveSearch?.({
        results: [serverHit({ session_id: 'server-only-2', snippet: 'a >>>Precious<<< relic', title: 'Relic notes' })]
      })
    })

    expect(await screen.findByText('Relic notes', {}, { timeout: 3000 })).toBeTruthy()

    const items = screen.getAllByRole('listitem').map(item => item.textContent ?? '')
    const localIndex = items.findIndex(text => text.includes('Precious conversation'))
    const serverIndex = items.findIndex(text => text.includes('Relic notes'))
    expect(localIndex).toBeGreaterThanOrEqual(0)
    expect(serverIndex).toBeGreaterThanOrEqual(0)
    expect(localIndex).toBeLessThan(serverIndex)
  })

  it('keeps the loaded match when the server search fails', async () => {
    // The endpoint is a second opinion, not the only one: a failed request must
    // leave the instant client-side match on screen (parity with the sidebar,
    // which also degrades to what it already has).
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {})

    mocks.searchSessions.mockRejectedValue(new Error('search endpoint unavailable'))
    renderCommandCenter()

    await typeSearch('Precious')

    await waitFor(() => expect(mocks.searchSessions).toHaveBeenCalledWith('Precious'), { timeout: 3000 })
    expect(await screen.findByText('Precious conversation', {}, { timeout: 3000 })).toBeTruthy()

    consoleError.mockRestore()
  })

  it('leaves the panel on the no-results copy, not the pending one, when the server search fails', async () => {
    // A rejected request must settle: the pending copy is a claim that an answer
    // is still coming, so leaving it up strands the panel on a spinner forever.
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {})

    mocks.searchSessions.mockRejectedValue(new Error('search endpoint unavailable'))
    renderCommandCenter()

    await typeSearch('zzqfailure')

    await waitFor(() => expect(mocks.searchSessions).toHaveBeenCalledWith('zzqfailure'), { timeout: 3000 })
    expect(await screen.findByText('No matching results found.', {}, { timeout: 3000 })).toBeTruthy()
    expect(screen.queryByText('Searching…')).toBeNull()

    consoleError.mockRestore()
  })

  it('shows the searching state while the request is unresolved, then the hit', async () => {
    // A query nothing loaded can match: the list is empty, so the pending
    // state is what fills the panel until the server answers.
    let resolveSearch: ((value: { results: SessionSearchResult[] }) => void) | undefined

    mocks.searchSessions.mockImplementation(
      () =>
        new Promise<{ results: SessionSearchResult[] }>(resolve => {
          resolveSearch = resolve
        })
    )
    renderCommandCenter()

    await typeSearch('zzqnomatch')

    await waitFor(() => expect(mocks.searchSessions).toHaveBeenCalledWith('zzqnomatch'), { timeout: 3000 })
    expect(await screen.findByText('Searching…', {}, { timeout: 3000 })).toBeTruthy()

    await act(async () => {
      resolveSearch?.({
        results: [serverHit({ session_id: 'server-late-1', snippet: 'a >>>zzqnomatch<<< relic', title: 'Late arrival' })]
      })
    })

    expect(await screen.findByText('Late arrival', {}, { timeout: 3000 })).toBeTruthy()
    expect(screen.queryByText('Searching…')).toBeNull()
  })
})

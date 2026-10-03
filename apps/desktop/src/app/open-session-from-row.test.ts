/**
 * `openSessionFromRow` is the one owner-aware door every list surface opens a
 * stored session through. Its contract: the ROW decides the owner, so the
 * resume goes to the backend that actually holds that row's session.
 *
 * Stored ids are only unique per profile (#92454), so two rows can share one
 * id; a session-scoped RPC only means anything on the backend that owns the
 * session's profile, so resolving by id alone dials the ambient backend and
 * the transcript never loads (#82527). These tests assert the RELATION between
 * the row and the routing the RPC dispatcher reads.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

vi.mock('@/store/session-states', () => ({
  canOpenSessionWindow: () => false,
  focusedSessionNeedsRoute: () => false,
  focusedSessionWorkspaceScope: () => ({ workspaceMode: 'sessions' }),
  focusOpenSession: () => null,
  frontMainIfSelected: () => undefined,
  openSessionTile: () => undefined,
  reuseBlankDraftTile: () => false,
  setSessionTileWorkspaceScope: () => undefined
}))

const row = (over: Partial<SessionInfo> = {}): SessionInfo => ({ id: 'stored-1', ...over }) as SessionInfo

const { $sessionResumeRequest, forgetSessionOwnerHintsForSession, requestSessionResume, setSessionOwnerHint } =
  await vi.hoisted(() => ({
    $sessionResumeRequest: { value: null as unknown },
    forgetSessionOwnerHintsForSession: vi.fn(),
    requestSessionResume: vi.fn(),
    setSessionOwnerHint: vi.fn()
  }))

vi.mock('@/store/session', async importOriginal => {
  const actual = (await importOriginal()) as Record<string, unknown>

  return {
    ...actual,
    $sessionResumeRequest: {
      get: () => $sessionResumeRequest.value,
      set: (next: unknown) => {
        $sessionResumeRequest.value = next
      },
      subscribe: () => () => undefined
    },
    forgetSessionOwnerHintsForSession,
    requestSessionResume,
    setSessionOwnerHint
  }
})

const { openSessionFromRow } = await import('../app/open-session')
// The REAL owner resolver (the mock factory spreads the actual module), so the
// expectation is derived from the one policy both surfaces share.
const { sessionOwnerRouteFromRow } = await import('@/store/session')

const navigate = vi.fn()

/** The expected owner, derived from the one shared resolver rather than
 *  restated — this test asserts the door applies the row's owner, not what the
 *  owner's field values happen to be. */
const ownerFor = (r: SessionInfo) => sessionOwnerRouteFromRow(r)

const resumeFor = (id: string) => {
  const calls = requestSessionResume.mock.calls.filter(([target]) => target === id)

  return calls[calls.length - 1]?.[1]
}

beforeEach(() => {
  requestSessionResume.mockClear()
  forgetSessionOwnerHintsForSession.mockClear()
  setSessionOwnerHint.mockClear()
  navigate.mockClear()
  $sessionResumeRequest.value = null
})

afterEach(() => {
  vi.clearAllMocks()
})

describe('openSessionFromRow routes the resume to the row that was picked', () => {
  it('pins the row’s own connection and profile as the resume owner', () => {
    const picked = row({ connection_id: 'source-b', profile: 'beta' })

    openSessionFromRow(picked, navigate)

    expect(resumeFor('stored-1')).toEqual(ownerFor(picked))
  })

  it('routes two rows sharing one stored id to their own backends', () => {
    const first = row({ connection_id: 'source-a', profile: 'alpha' })
    const second = row({ connection_id: 'source-b', profile: 'beta' })

    openSessionFromRow(first, navigate)
    openSessionFromRow(second, navigate)

    // The second open must not inherit the first row's owner: these are two
    // different conversations that happen to share a stored id.
    expect(resumeFor('stored-1')).toEqual(ownerFor(second))
    expect(ownerFor(second)).not.toEqual(ownerFor(first))
  })

  it('drops a stale explicit owner for an untagged row instead of honoring it', () => {
    // A row with no connection tag belongs to whichever backend returned it, so
    // a leftover hint from an earlier owner would retarget the resume.
    openSessionFromRow(row({ profile: 'default' }), navigate)

    expect(forgetSessionOwnerHintsForSession).toHaveBeenCalledWith('stored-1')
    expect(resumeFor('stored-1')).toBeUndefined()
  })

  it('treats an absent connection tag as untagged rather than as the local one', () => {
    openSessionFromRow(row({ connection_id: '   ', profile: 'default' }), navigate)

    expect(forgetSessionOwnerHintsForSession).toHaveBeenCalledWith('stored-1')
  })
})

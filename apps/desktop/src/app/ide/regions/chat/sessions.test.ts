// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from 'vitest'

const listMock = vi.fn()
const openTileMock = vi.fn()
const patchTileMock = vi.fn()

vi.mock('@/api/sessions', () => ({
  listAllProfileSessions: (...args: unknown[]) => listMock(...args)
}))

vi.mock('@/store/session-states', () => ({
  openSessionTile: (...args: unknown[]) => openTileMock(...args),
  patchSessionTile: (...args: unknown[]) => patchTileMock(...args)
}))

import { $activeGatewayProfile } from '@/store/profile'
import { $sessions } from '@/store/session'

import { $ideWorkspaceRoot } from '../../state'

import { createIdeSession, listIdeSessions, rememberIdeSessionRows, rememberIdeSessionTitle } from './sessions'
import { $ideActiveChat } from './store'

beforeEach(() => {
  listMock.mockReset()
  openTileMock.mockReset()
  $ideActiveChat.set(null)
  $ideWorkspaceRoot.set(null)
  $activeGatewayProfile.set('work')
  $sessions.set([])
})

describe('createIdeSession', () => {
  it('creates a source=ide session with the IDE workspace and profile', async () => {
    $ideWorkspaceRoot.set('D:/repo/one')

    const request = vi.fn().mockResolvedValue({ stored_session_id: 's1' })
    const stored = await createIdeSession(request)

    expect(stored).toBe('s1')
    expect(request).toHaveBeenCalledWith('session.create', {
      cols: 96,
      cwd: 'D:/repo/one',
      profile: 'work',
      source: 'ide'
    })
    expect(openTileMock).toHaveBeenCalledWith('s1')
    expect($ideActiveChat.get()).toBe('s1')
  })

  it('omits cwd when the IDE has no workspace', async () => {
    const request = vi.fn().mockResolvedValue({ stored_session_id: 's2' })

    await createIdeSession(request)

    expect(request.mock.calls[0][1]).toEqual({ cols: 96, profile: 'work', source: 'ide' })
  })

  it('returns null and opens nothing when no stored id came back', async () => {
    const request = vi.fn().mockResolvedValue({})

    expect(await createIdeSession(request)).toBe(null)
    expect(openTileMock).not.toHaveBeenCalled()
    expect($ideActiveChat.get()).toBe(null)
  })
})

describe('listIdeSessions', () => {
  it('queries the ide slice through the REST source filter', async () => {
    listMock.mockResolvedValue({ sessions: [{ id: 's1' }] })

    const rows = await listIdeSessions()

    expect(listMock).toHaveBeenCalledWith(50, 0, 'exclude', 'recent', 'work', { source: 'ide' })
    expect(rows).toEqual([{ id: 's1' }])
  })
})

describe('rememberIdeSessionRows', () => {
  it('upserts unknown rows so tab titles resolve', () => {
    $sessions.set([{ id: 'old' } as never])

    rememberIdeSessionRows([{ id: 'old' } as never, { id: 'new' } as never])

    expect($sessions.get().map(session => session.id).sort()).toEqual(['new', 'old'])
  })

  it('no-ops on an empty list', () => {
    $sessions.set([{ id: 'old' } as never])

    rememberIdeSessionRows([])

    expect($sessions.get().map(session => session.id)).toEqual(['old'])
  })
})

describe('rememberIdeSessionTitle', () => {
  it('inserts a titled ide row when the session is unknown', () => {
    rememberIdeSessionTitle('s1', 'Fresh title')

    const rows = $sessions.get()

    expect(rows).toHaveLength(1)
    expect(rows[0].id).toBe('s1')
    expect(rows[0].title).toBe('Fresh title')
    expect(rows[0].source).toBe('ide')
  })

  it('retitles an existing row in place', () => {
    $sessions.set([{ id: 's1', title: 'New session' } as never])

    rememberIdeSessionTitle('s1', 'Renamed by titler')

    expect($sessions.get()[0].title).toBe('Renamed by titler')
  })

  it('no-ops (identity preserved) when the title is unchanged', () => {
    const row = { id: 's1', title: 'Same' } as never

    $sessions.set([row])

    rememberIdeSessionTitle('s1', 'Same')

    expect($sessions.get()[0]).toBe(row)
  })
})

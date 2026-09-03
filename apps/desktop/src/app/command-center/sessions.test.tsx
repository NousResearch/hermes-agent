import { act, renderHook } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'

import {
  commandCenterSessionOwnerRoute,
  mergeCommandCenterSessions,
  openCommandCenterSession,
  useCommandCenterSessions
} from './sessions'

const listAllProfileSessions = vi.hoisted(() => vi.fn())
const openSession = vi.hoisted(() => vi.fn())
const requestSessionResume = vi.hoisted(() => vi.fn())
const setSessionOwnerHint = vi.hoisted(() => vi.fn())

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  listAllProfileSessions: (...args: unknown[]) => listAllProfileSessions(...args)
}))

vi.mock('../open-session', () => ({ openSession: (...args: unknown[]) => openSession(...args) }))

vi.mock('@/store/session', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestSessionResume: (...args: unknown[]) => requestSessionResume(...args),
  setSessionOwnerHint: (...args: unknown[]) => setSessionOwnerHint(...args)
}))

const row = (id: string, profile: string, over: Partial<SessionInfo> = {}): SessionInfo =>
  ({
    ended_at: null,
    id,
    input_tokens: 0,
    is_active: false,
    last_active: 100,
    message_count: 1,
    model: null,
    output_tokens: 0,
    preview: null,
    profile,
    source: 'desktop',
    started_at: 90,
    title: id,
    tool_call_count: 0,
    ...over
  }) as SessionInfo

describe('command center session aggregation', () => {
  beforeEach(() => {
    listAllProfileSessions.mockReset()
    openSession.mockReset()
    requestSessionResume.mockReset()
    setSessionOwnerHint.mockReset()
  })

  it('keeps profiles distinct and dedupes the same durable session identity deterministically', () => {
    const sessions = mergeCommandCenterSessions(
      [],
      [
        row('shared', 'clientops', { last_active: 100, title: 'stale' }),
        row('shared', 'agentops', { last_active: 150, title: 'other profile' }),
        row('shared', 'clientops', { last_active: 200, title: 'fresh' }),
        row('shared', 'clientops', { connection_id: 'remote-a', last_active: 175, title: 'remote twin' })
      ]
    )

    expect(
      sessions.map(session => [session.connection_id ?? 'local', session.profile, session.id, session.title])
    ).toEqual([
      ['local', 'clientops', 'shared', 'fresh'],
      ['remote-a', 'clientops', 'shared', 'remote twin'],
      ['local', 'agentops', 'shared', 'other profile']
    ])
  })

  it('carries the last known rows for a profile whose database read failed', () => {
    const sessions = mergeCommandCenterSessions(
      [row('worker', 'clientops', { last_active: 200 }), row('old-default', 'default')],
      [row('new-default', 'default', { last_active: 300 })],
      [{ error: 'unable to open database file', profile: 'clientops' }]
    )

    expect(sessions.map(session => `${session.profile}:${session.id}`)).toEqual([
      'default:new-default',
      'clientops:worker'
    ])
  })

  it('carries failed rows for only the exact connection and profile owner', () => {
    const sessions = mergeCommandCenterSessions(
      [
        row('failed-worker', 'clientops', { connection_id: 'remote-a', last_active: 200 }),
        row('healthy-stale', 'clientops', { connection_id: 'remote-b', last_active: 190 })
      ],
      [row('healthy-fresh', 'clientops', { connection_id: 'remote-b', last_active: 300 })],
      [{ connection_id: 'remote-a', error: 'gateway unavailable', profile: 'clientops' }]
    )

    expect(sessions.map(session => `${session.connection_id}:${session.id}`)).toEqual([
      'remote-b:healthy-fresh',
      'remote-a:failed-worker'
    ])
  })

  it('carries every prior profile when a shared gateway aggregate fails', () => {
    const sessions = mergeCommandCenterSessions(
      [
        row('client-worker', 'clientops', { connection_id: 'remote-a' }),
        row('agent-worker', 'agentops', { connection_id: 'remote-a' }),
        row('stale-other', 'clientops', { connection_id: 'remote-b' })
      ],
      [row('fresh-other', 'clientops', { connection_id: 'remote-b', last_active: 300 })],
      [{ connection_id: 'remote-a', error: 'gateway unavailable', profile: 'all' }]
    )

    expect(sessions.map(session => `${session.connection_id}:${session.id}`).sort()).toEqual([
      'remote-a:agent-worker',
      'remote-a:client-worker',
      'remote-b:fresh-other'
    ])
  })

  it('ignores a stale cross-profile response that lands after a newer refresh', async () => {
    let resolveOld!: (value: unknown) => void
    let resolveNew!: (value: unknown) => void

    const oldResponse = new Promise(resolve => {
      resolveOld = resolve
    })

    const newResponse = new Promise(resolve => {
      resolveNew = resolve
    })

    listAllProfileSessions.mockReturnValueOnce(oldResponse).mockReturnValueOnce(newResponse)

    const { result } = renderHook(() => useCommandCenterSessions(true))

    await vi.waitFor(() => expect(listAllProfileSessions).toHaveBeenCalledTimes(1))
    let refresh!: Promise<void>
    act(() => {
      refresh = result.current.refresh()
    })

    await act(async () => {
      resolveNew({ errors: [], sessions: [row('new', 'clientops')], total: 1 })
      await refresh
    })
    expect(result.current.sessions.map(session => session.id)).toEqual(['new'])

    await act(async () => {
      resolveOld({ errors: [], sessions: [row('stale', 'agentops')], total: 1 })
      await oldResponse
    })
    expect(result.current.sessions.map(session => session.id)).toEqual(['new'])
    expect(listAllProfileSessions).toHaveBeenLastCalledWith(500, 0, 'include', 'recent', 'all')
  })

  it('routes a selected row through its exact profile socket', () => {
    expect(commandCenterSessionOwnerRoute(row('local-worker', 'clientops'))).toEqual({
      connectionId: 'local',
      profile: 'clientops'
    })
    expect(commandCenterSessionOwnerRoute(row('remote-worker', 'agentops', { connection_id: 'remote-a' }))).toEqual({
      connectionId: 'remote-a',
      profile: 'agentops'
    })
  })

  it('forces an exact-owner main resume instead of focusing a same-id tile from another owner', () => {
    const navigate = vi.fn()
    const session = row('shared', 'agentops', { connection_id: 'remote-a' })

    openCommandCenterSession(session, navigate)

    const owner = { connectionId: 'remote-a', profile: 'agentops' }

    expect(setSessionOwnerHint).toHaveBeenCalledWith('shared', owner)
    expect(openSession).toHaveBeenCalledWith('shared', navigate, 'main')
    expect(requestSessionResume).toHaveBeenCalledWith('shared', owner)
  })
})

/**
 * Duplicating a bot: clone the profile (config/skills/SOUL/memory via
 * `clone_from`) and copy the LOOK, but never the things that belong to the
 * original — its canonical-chat pointer and its creation stamp.
 *
 * The name search is the other half. Candidates are `<base>-2`, `-3`, … and
 * the BASE is truncated to fit, never the suffix (#19): slicing the joined
 * string chops the "-2" off a max-length name, so the candidate collides with
 * the base forever and the search runs out at -99.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $botMeta } from './data'
import { duplicateBot, duplicateNameTaken, nextDuplicateName } from './profile-ops'
import type { RosterRow } from './types'

const { ensureBotMetadataMock, faceOnlyMock, hostMock, storageMock } = vi.hoisted(() => ({
  ensureBotMetadataMock: vi.fn(),
  faceOnlyMock: vi.fn((_data: string) => false),
  hostMock: {
    request: vi.fn(),
    requestProfile: vi.fn(),
    state: { connectionId: { get: () => 'local' }, focusedSessionOwner: null, profile: { get: () => 'default' } }
  },
  storageMock: { get: vi.fn(), set: vi.fn() }
}))

vi.mock('@hermes/plugin-sdk', async () => {
  const { atom } = await import('nanostores')

  return {
    atom,
    forgetSessionUnread: vi.fn(),
    host: hostMock,
    queryClient: { invalidateQueries: vi.fn() },
    useQuery: vi.fn(),
    useValue: vi.fn()
  }
})

vi.mock('./shared', () => ({ getPluginCtx: () => ({ storage: storageMock }), ID: 'hermes-bots' }))
vi.mock('./avatar-image', () => ({ isBackfilledFacePng: (data: string) => faceOnlyMock(data) }))
vi.mock('./canonical-chat', () => ({ ensureBotMetadata: ensureBotMetadataMock }))

const calls: Array<{ method: string; params: Record<string, unknown> }> = []

beforeEach(() => {
  vi.clearAllMocks()
  faceOnlyMock.mockReturnValue(false)
  calls.length = 0
  $botMeta.set({})
  storageMock.set.mockResolvedValue(undefined)
  hostMock.request.mockImplementation(async (method: string, params: Record<string, unknown>) => {
    calls.push({ method, params: structuredClone(params ?? {}) })

    return { ok: true }
  })
})

describe('duplicating a bot', () => {
  it('copies the look but neither the chat pointer nor the creation stamp', async () => {
    $botMeta.set({
      researcher: {
        chat: 'sess-source-forever',
        color: '#f97316',
        created: 1_700_000_000_000,
        image: 'data:image/png;base64,xx',
        shape: 'circle',
        title: 'Researcher'
      }
    })

    const name = await duplicateBot({ description: 'finds things', name: 'researcher' } as RosterRow, [
      { name: 'researcher' } as RosterRow
    ])

    expect(name).toBe('researcher-2')

    const clone = $botMeta.get()['researcher-2']

    expect(clone).toMatchObject({
      color: '#f97316',
      image: 'data:image/png;base64,xx',
      shape: 'circle',
      title: 'Researcher (copy)'
    })
    expect(clone.chat).toBeUndefined()
    expect(clone.created).toBeUndefined()

    expect(calls.find(call => call.method === 'profiles.create')?.params).toMatchObject({
      clone_from: 'researcher',
      name: 'researcher-2'
    })

    const configure = calls.filter(call => call.method === 'profiles.configure').at(-1)
    const uiMeta = (configure?.params.ui_meta as Record<string, Record<string, unknown>>)['hermes-bots']

    expect(uiMeta.title).toBe('Researcher (copy)')
    expect(uiMeta.chat).toBeUndefined()
    expect(uiMeta.created).toBeUndefined()
  })

  it('duplicates the look of a bot that never had a pointer', async () => {
    $botMeta.set({ painter: { color: '#38bdf8', shape: 'cloud', title: 'Painter' } })

    const name = await duplicateBot({ name: 'painter' } as RosterRow, [{ name: 'painter' } as RosterRow])

    expect($botMeta.get()[name]).toMatchObject({ shape: 'cloud', title: 'Painter (copy)' })
    expect($botMeta.get()[name].chat).toBeUndefined()
  })

  it('walks past taken suffixes to the first free slot', async () => {
    const roster = ['ops', 'ops-2', 'ops-3'].map(name => ({ name }) as RosterRow)

    expect(await duplicateBot({ name: 'ops' } as RosterRow, roster)).toBe('ops-4')
  })

  it('uses a caller-chosen name and refuses a taken one instead of re-suffixing', async () => {
    const roster = ['ops', 'ops-2'].map(name => ({ name }) as RosterRow)

    // A free chosen name wins verbatim — no suffix is appended to it.
    expect(await duplicateBot({ name: 'ops' } as RosterRow, roster, { name: 'night-crew' })).toBe('night-crew')

    // A taken chosen name is an error the dialog can surface, never a silent
    // fallback to an auto-suffix the user did not ask for.
    await expect(duplicateBot({ name: 'ops' } as RosterRow, roster, { name: 'ops-2' })).rejects.toThrow(
      /already taken/
    )
  })

  it('truncates the BASE so a max-length name still gets a distinct suffix (#19)', async () => {
    const base = 'b'.repeat(64)

    const name = await duplicateBot({ name: base } as RosterRow, [{ name: base } as RosterRow])

    expect(name).toHaveLength(64)
    expect(name.endsWith('-2')).toBe(true)
    expect(name).not.toBe(base)
  })

  it('ensures the source bot has its metadata before cloning', async () => {
    // clone_from copies the profile dir; the source's Bot Chat has to exist
    // first or the clone inherits a half-built profile.
    await duplicateBot({ name: 'ops' } as RosterRow, [])

    expect(ensureBotMetadataMock).toHaveBeenCalledTimes(1)
  })

  it('only collides against rows on the SAME connection', async () => {
    // A same-named bot on another gateway is a different agent entirely.
    const bot = {
      connectionId: 'vera',
      name: 'ops',
      route: { connectionId: 'vera', mode: 'remote', profile: 'ops', targetProfile: 'ops' },
      sourceScoped: true
    } as RosterRow

    const elsewhere = {
      connectionId: 'other',
      name: 'ops-2',
      route: { connectionId: 'other', mode: 'remote', profile: 'ops-2', targetProfile: 'ops-2' },
      sourceScoped: true
    } as RosterRow

    hostMock.requestProfile.mockResolvedValue({ ok: true })

    expect(await duplicateBot(bot, [bot, elsewhere])).toBe('ops-2')
  })
})

/**
 * Roster avatar sync must stay on the ACTIVE gateway (#102978). Every row of
 * a local-primary roster is source-scoped, so routing `profiles.get_asset` /
 * `set_asset` per row through the (connectionId, profile) secondary dialed a
 * pooled backend for every registered profile on the first paint after
 * launch — 60 profiles against 3 slots, a queue that never drained. The
 * active gateway's `profiles.list` already read those directories; the asset
 * RPCs are the same reads, addressed by the row's backend name.
 */
describe('roster avatar sync (#102978)', () => {
  it('fetches has_avatar art through the active gateway, never a per-row secondary dial', async () => {
    const { pullServerAvatars } = await import('./profile-ops')
    hostMock.request.mockResolvedValue({ found: false })

    const roster = ['alpha', 'beta', 'gamma'].map(
      name =>
        ({
          connectionId: 'local',
          connectionKind: 'local',
          has_avatar: true,
          name,
          route: { connectionId: 'local', mode: 'local', profile: name, targetProfile: name },
          sourceScoped: true
        }) as RosterRow
    )

    pullServerAvatars(roster)
    await Promise.resolve()

    expect(hostMock.requestProfile).not.toHaveBeenCalled()
    expect(hostMock.request.mock.calls.map(([method, params]) => [method, params.name])).toEqual([
      ['profiles.get_asset', 'alpha'],
      ['profiles.get_asset', 'beta'],
      ['profiles.get_asset', 'gamma']
    ])
  })

  it('addresses an aliased row by its backend profile name', async () => {
    const { pullServerAvatars } = await import('./profile-ops')
    hostMock.request.mockResolvedValue({ found: false })

    pullServerAvatars([
      {
        connectionId: 'local',
        has_avatar: true,
        name: 'mara',
        route: { connectionId: 'local', mode: 'local', profile: 'mara', targetProfile: 'default' },
        sourceScoped: true
      } as RosterRow
    ])
    await Promise.resolve()

    expect(hostMock.requestProfile).not.toHaveBeenCalled()
    expect(hostMock.request).toHaveBeenCalledWith('profiles.get_asset', { asset: 'avatar', name: 'default' })
  })

  it('does not re-fetch a face-only raster on the next roster tick (#99336)', async () => {
    const { pullServerAvatars } = await import('./profile-ops')
    faceOnlyMock.mockReturnValue(true)
    hostMock.request.mockResolvedValue({ data: 'data:image/png;base64,AAAA', found: true })

    const row = {
      connectionId: 'local',
      has_avatar: true,
      name: 'secretary',
      route: { connectionId: 'local', mode: 'local', profile: 'secretary', targetProfile: 'secretary' },
      sourceScoped: true
    } as RosterRow

    pullServerAvatars([row])
    await vi.waitFor(() => expect(hostMock.request).toHaveBeenCalledTimes(1))
    // Let the first fetch settle (its in-flight guard clears in a finally) so
    // the second tick is decided by the memo, not by that guard.
    await new Promise(resolve => setTimeout(resolve, 0))
    pullServerAvatars([row])
    await new Promise(resolve => setTimeout(resolve, 0))

    expect(hostMock.request).toHaveBeenCalledTimes(1)
    // The raster is a notice-only copy of the live face, never parked on the roster.
    expect($botMeta.get()['local::secretary']?.image).toBeUndefined()
  })
})

describe('duplicate naming', () => {
  it('truncates the BASE, never the suffix (#19)', () => {
    const base = 'b'.repeat(64)
    const name = nextDuplicateName({ name: base } as RosterRow, [{ name: base } as RosterRow])

    // Slicing the joined string would chop the "-2" off a max-length name, so
    // every candidate collapses back onto the base and collides with it forever.
    expect(name).toBe(`${'b'.repeat(62)}-2`)
    expect(name!.length).toBeLessThanOrEqual(64)
  })

  it('judges a name against the bot connection, not the whole roster', () => {
    // A bot on another connection: the operation mints into ITS connection, so a
    // name taken on a different one is free here — and the dialog must agree.
    const bot = { connectionId: 'remote-a', name: 'researcher', remoteSource: true } as RosterRow
    const elsewhere = { connectionId: 'remote-b', name: 'researcher-2', remoteSource: true } as RosterRow
    const sameConnection = { connectionId: 'remote-a', name: 'researcher-2', remoteSource: true } as RosterRow

    expect(duplicateNameTaken(bot, [elsewhere], 'researcher-2')).toBe(false)
    expect(nextDuplicateName(bot, [elsewhere])).toBe('researcher-2')

    // Taken on the bot's OWN connection, refused.
    expect(duplicateNameTaken(bot, [sameConnection], 'researcher-2')).toBe(true)
  })

  it('stays name-wide for a local bot, matching the operation', () => {
    // A local bot has no owner route, so `duplicateBot` treats any row with that
    // name as taken. The dialog must not be stricter than the thing it previews.
    const bot = { name: 'researcher' } as RosterRow

    expect(duplicateNameTaken(bot, [{ name: 'researcher-2' } as RosterRow], 'researcher-2')).toBe(true)
  })
})

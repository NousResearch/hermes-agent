/**
 * Cross-connection routing: how a roster row resolves its owning connection,
 * how RPCs ride that owner, and how a configured Desktop alias keeps its
 * friendly identity after the backend it points at answers the roster.
 *
 * #89131 is the alias half: a per-profile Cloud alias ("moxie" → exact Cloud
 * connection → backend targetProfile "default") lost its name the moment the
 * hosted session activated, because the row that came back was keyed
 * (cloud-abc, default) — a different identity than the alias meta. It read as
 * the raw Cloud hostname, or as generic "Hermes" when Cloud was the only
 * source.
 *
 * Drives the real `routing` + `labels` modules; only the SDK `host` is mocked.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

import { displayName } from './labels'
import {
  aliasIdentityFor,
  beginAliasRouteIndex,
  botConnectionRoute,
  botRosterMeta,
  indexAliasRoutes,
  requestForBot,
  resolveBotConnectionRoute
} from './routing'
import type { ProfileRoute, RosterRow } from './types'

const { hostMock } = vi.hoisted(() => ({
  hostMock: {
    request: vi.fn(),
    requestProfile: vi.fn(),
    state: { connectionId: { get: vi.fn(() => 'local') } }
  }
}))

vi.mock('@hermes/plugin-sdk', () => ({ host: hostMock }))

const MOXIE_ROUTE: ProfileRoute = {
  connectionId: 'cloud-abc',
  mode: 'remote',
  profile: 'moxie',
  targetProfile: 'default'
}

/** The hosted backend's own roster row after alias handoff: the Cloud
 *  connection answers as its root profile, so the row identity is
 *  (cloud-abc, default) — NOT the alias key. */
const hostedRow = {
  connectionId: 'cloud-abc',
  connectionLabel: 'cloud.example.com',
  name: 'default',
  remoteSource: true,
  route: { connectionId: 'cloud-abc', mode: 'remote', profile: 'default', targetProfile: 'default' },
  sourceScoped: true,
  targetProfile: 'default'
} as RosterRow

beforeEach(() => {
  vi.clearAllMocks()
  hostMock.state.connectionId.get.mockReturnValue('local')
  indexAliasRoutes([])
})

describe('alias identity survives the hosted handoff (#89131)', () => {
  it('names the backend row after the alias, through every meta generation', () => {
    indexAliasRoutes([
      { connectionId: 'local', mode: 'local', profile: 'default', targetProfile: 'default' },
      MOXIE_ROUTE
    ])

    // No stored title anywhere: the alias NAME is still the identity.
    expect(displayName(hostedRow, null)).toBe('Moxie')

    // The alias's Bot Mode title claims the row under either meta key shape —
    // aliases predate the v2 (connection-qualified) migration on mixed setups.
    const metaV2 = { 'cloud-abc::moxie': { title: 'Moxie' } }

    expect(botRosterMeta(hostedRow, metaV2)).toBe(metaV2['cloud-abc::moxie'])
    expect(displayName(hostedRow, botRosterMeta(hostedRow, metaV2))).toBe('Moxie')

    const metaV1 = { moxie: { title: 'Moxie ✨' } }

    expect(displayName(hostedRow, botRosterMeta(hostedRow, metaV1))).toBe('Moxie ✨')
  })

  it('renders the sole Cloud-only default as the alias, not "Hermes"', () => {
    // Global route is Cloud: the active gateway IS the Cloud connection and
    // profiles.list returns one unannotated rich `default` row.
    hostMock.state.connectionId.get.mockReturnValue('cloud-abc')
    indexAliasRoutes([MOXIE_ROUTE])

    expect(displayName({ name: 'default' }, null)).toBe('Moxie')
    // A user-set title still wins over the alias name.
    expect(displayName({ name: 'default' }, { title: 'Custom' })).toBe('Custom')
  })

  it('never leaks the alias onto a same-named default on another connection', () => {
    indexAliasRoutes([MOXIE_ROUTE])

    const otherDefault = {
      ...hostedRow,
      connectionId: 'other-conn',
      connectionLabel: 'Personal',
      route: undefined
    } as RosterRow

    expect(aliasIdentityFor(otherDefault)).toBeNull()
    expect(displayName(otherDefault, null)).toBe('Personal')
    // Local default while the ACTIVE gateway is local: untouched "Hermes".
    expect(displayName({ name: 'default' }, null)).toBe('Hermes')
  })

  it('fails closed when two aliases claim one backend row', () => {
    indexAliasRoutes([
      MOXIE_ROUTE,
      { connectionId: 'cloud-abc', mode: 'remote', profile: 'roxie', targetProfile: 'default' }
    ])

    expect(aliasIdentityFor(hostedRow)).toBeNull()
    // Ambiguous: fall back to the source label, not a guessed alias.
    expect(displayName(hostedRow, null)).toBe('cloud.example.com')
  })

  it('never lets the alias row claim its own alias entry', () => {
    indexAliasRoutes([MOXIE_ROUTE])

    const aliasRow = {
      connectionId: 'cloud-abc',
      name: 'moxie',
      route: { ...MOXIE_ROUTE },
      sourceScoped: true,
      targetProfile: 'default'
    } as RosterRow

    expect(aliasIdentityFor(aliasRow)).toBeNull()
    expect(displayName(aliasRow, null)).toBe('Moxie')
  })

  it('keeps only genuine aliases, and drops stale claims on refresh', () => {
    indexAliasRoutes([{ connectionId: 'local', mode: 'local', profile: 'rune', targetProfile: 'rune' }, MOXIE_ROUTE])

    expect(aliasIdentityFor(hostedRow)).toBeTruthy()

    // Alias removed from config → next inventory drops the claim.
    indexAliasRoutes([{ connectionId: 'local', mode: 'local', profile: 'rune', targetProfile: 'rune' }])

    expect(aliasIdentityFor(hostedRow)).toBeNull()
    expect(displayName(hostedRow, null)).toBe('cloud.example.com')
  })
})

describe('overlapping index rebuilds', () => {
  const deferred = () => {
    let resolve!: (routes: ProfileRoute[]) => void

    const promise = new Promise<ProfileRoute[]>(settle => {
      resolve = settle
    })

    return { promise, resolve }
  }

  /** The rebuild exactly as useRoster runs it: claim the epoch, read the
   *  route inventory over the wire, then replace the index wholesale. */
  const rebuild = async (read: Promise<ProfileRoute[]>) => {
    const epoch = beginAliasRouteIndex()

    indexAliasRoutes(await read, epoch)
  }

  const STALE_ROUTE: ProfileRoute = {
    connectionId: 'cloud-abc',
    mode: 'remote',
    profile: 'stale',
    targetProfile: 'default'
  }

  it('keeps the newer routes when an older read resolves last', async () => {
    const older = deferred()
    const newer = deferred()

    const first = rebuild(older.promise)
    const second = rebuild(newer.promise)

    newer.resolve([MOXIE_ROUTE])
    await second

    older.resolve([STALE_ROUTE])
    await first

    expect(aliasIdentityFor(hostedRow)?.name).toBe('moxie')
  })

  it('still adopts the newest routes once rebuilds stop overlapping', async () => {
    await rebuild(Promise.resolve([MOXIE_ROUTE]))

    expect(aliasIdentityFor(hostedRow)?.name).toBe('moxie')

    await rebuild(Promise.resolve([STALE_ROUTE]))

    expect(aliasIdentityFor(hostedRow)?.name).toBe('stale')
  })
})

describe('a row without a reachable owner', () => {
  it('reports a typed status instead of throwing, while dispatch still fails closed', () => {
    expect(resolveBotConnectionRoute({ name: 'plain' })).toEqual({ route: null, status: 'not_scoped' })
    expect(resolveBotConnectionRoute({ connectionId: 'vera', name: 'ops', sourceScoped: true })).toMatchObject({
      route: { connectionId: 'vera', mode: 'remote', profile: 'ops', targetProfile: 'ops' },
      status: 'resolved'
    })

    const orphan = { name: 'ops', remoteSource: true } as RosterRow

    expect(resolveBotConnectionRoute(orphan)).toMatchObject({ profile: 'ops', route: null, status: 'owner_removed' })
    // The strict wrapper real dispatch uses must still refuse — a row whose
    // connection was deleted has no ambient fallback to borrow.
    expect(() => botConnectionRoute(orphan)).toThrow(/no connection owner/)
  })

  it('a passive meta lookup on an orphaned row reads as "no route", never a throw', () => {
    const orphan = { name: 'ops', remoteSource: true } as RosterRow

    expect(botRosterMeta(orphan, { ops: { title: 'Ops' } })).toBeFalsy()
  })
})

describe('requestForBot rides the bot’s own source', () => {
  it('pins same-name bots to their own connection under concurrent requests', async () => {
    hostMock.requestProfile.mockImplementation(async (route: ProfileRoute) => ({ from: route.connectionId }))

    const rows = ['vera', 'mac-mini'].map(
      connectionId =>
        ({
          connectionId,
          name: 'default',
          remoteSource: true,
          sourceScoped: true
        }) as RosterRow
    )

    const answers = await Promise.all(rows.map(bot => requestForBot<{ from: string }>(bot, 'profiles.list', {})))

    expect(answers.map(answer => answer.from)).toEqual(['vera', 'mac-mini'])
    expect(hostMock.request).not.toHaveBeenCalled()
  })

  it('rewrites the logical profile to the backend target on the wire', async () => {
    hostMock.requestProfile.mockResolvedValue({})

    await requestForBot({ name: 'moxie', route: MOXIE_ROUTE, sourceScoped: true } as RosterRow, 'profiles.configure', {
      name: 'moxie',
      soul: '# hi'
    })

    expect(hostMock.requestProfile).toHaveBeenCalledWith(MOXIE_ROUTE, 'profiles.configure', {
      name: 'default',
      soul: '# hi'
    })
  })

  it('fails closed rather than falling back to the ambient request', async () => {
    // A scoped row whose shell predates requestProfile must NOT silently
    // execute against whichever gateway happens to be active.
    const scoped = { connectionId: 'vera', name: 'ops', sourceScoped: true } as RosterRow

    hostMock.requestProfile = undefined as never

    await expect(requestForBot(scoped, 'profiles.list', {})).rejects.toThrow(/Cannot route profiles\.list/)
    expect(hostMock.request).not.toHaveBeenCalled()

    hostMock.requestProfile = vi.fn()
  })

  it('coerces a JSON-RPC rejection into an Error with a string name (#94471)', async () => {
    // React 19 formats query errors with `(error.name || '').trim()`; a
    // numeric JSON-RPC `name` crashed the Routines pane and hid the cause.
    hostMock.request.mockRejectedValue({ code: -32000, message: 'profile busy', name: -32000 })

    const error = await requestForBot({ name: 'ops' }, 'cron.list', {}).catch((thrown: unknown) => thrown)

    expect(error).toBeInstanceOf(Error)
    expect(typeof (error as Error).name).toBe('string')
    expect((error as Error).message).toBe('profile busy')
  })
})

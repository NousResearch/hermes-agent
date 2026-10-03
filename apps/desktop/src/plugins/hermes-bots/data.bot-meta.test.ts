/**
 * Bot metadata: the write path (`saveBotMeta`), the disk-load path
 * (`migrateBotMeta`), and the server-reconciliation overlay
 * (`mergeServerMeta`).
 *
 * Every case below is a shipped regression:
 *  - a save must report a REAL remote failure differently from the documented
 *    older-gateway fallback, or the editor toasts "remote persistence failed"
 *    on every save forever;
 *  - `profiles.set_asset` must fire only when the avatar actually changed — a
 *    no-op `clear` from one machine raced another machine's just-pushed avatar
 *    and wiped it server-side;
 *  - a slow disk load must not wipe metadata written earlier in the session;
 *  - the server overlay is authoritative for the fields it carries (a dropped
 *    `chat` pointer, a disbanded group) but must NOT resurrect state a local
 *    write already removed — the disband-resurrection fence.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $botMeta, $lastRoster, botMentionTag, botMetaWriteAt, mergeMultiSourceRoster, saveBotMeta } from './data'
import { editedLook } from './edit-profile-look'
import { groupSpeakerLabel } from './group-chat'
import { buildGroupChatTurnPrompt } from './group-round-prompt'
import { displayName } from './labels'
import { mergeServerMeta } from './profile-ops'
import { botRosterMeta } from './routing'
import type { RosterRow } from './types'

const { hostMock, storageMock } = vi.hoisted(() => ({
  hostMock: {
    agents: undefined as unknown,
    profileRoutes: undefined as unknown,
    request: vi.fn(),
    requestProfile: vi.fn(),
    state: { connectionId: { get: () => 'local' }, profile: { get: () => 'default' } }
  },
  storageMock: { get: vi.fn(), remove: vi.fn(), set: vi.fn() }
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

// profile-ops pulls in the roster surfaces for delete/duplicate; the overlay
// under test needs none of them.
vi.mock('./avatar-image', () => ({ isBackfilledFacePng: () => false }))
vi.mock('./canonical-chat', () => ({ ensureBotMetadata: vi.fn() }))

/** The backend's profile listing: a save reads the bot's server namespace
 *  before it writes, so every profile a case saves has to exist there. */
const SERVED_PROFILES = {
  profiles: ['beta', 'ops', 'researcher', 'source', 'source-2'].map(name => ({ name, ui_meta_revisions: {} }))
}

/** Every `host.request` recorded, with the params frozen at call time. */
function recordRequests(reply: (method: string) => unknown = () => ({})) {
  const calls: Array<{ method: string; params: Record<string, unknown> }> = []

  hostMock.request.mockImplementation(async (method: string, params: Record<string, unknown>) => {
    calls.push({ method, params: structuredClone(params ?? {}) })

    return method === 'profiles.list' ? SERVED_PROFILES : reply(method)
  })

  return calls
}

beforeEach(() => {
  vi.clearAllMocks()
  $botMeta.set({})
  botMetaWriteAt.clear()
  storageMock.get.mockResolvedValue(null)
  storageMock.set.mockResolvedValue(undefined)
  storageMock.remove.mockResolvedValue(undefined)
  recordRequests()
})

describe('a save reports what the SERVER did, not just what the screen shows', () => {
  it('keeps the look locally and reports the remote failure', async () => {
    recordRequests(method => (method === 'profiles.configure' ? { applied: { ui_meta: false } } : {}))

    const result = await saveBotMeta('researcher', { color: '#38bdf8', custom: true, shape: 'cloud' })

    expect($botMeta.get().researcher).toMatchObject({ color: '#38bdf8', shape: 'cloud' })
    expect(result).toEqual({ serverOutcome: 'failed', serverPersisted: false })
  })

  it('reports success once ui_meta is applied', async () => {
    recordRequests(method => (method === 'profiles.configure' ? { applied: { ui_meta: true } } : {}))

    await expect(saveBotMeta('researcher', { color: '#8b5cf6', custom: true, shape: 'hexagon' })).resolves.toEqual({
      serverOutcome: 'persisted',
      serverPersisted: true
    })
  })

  it('treats an older gateway with no `applied` contract as unsupported, not failed', async () => {
    // An error toast on this path would fire on EVERY save forever.
    recordRequests(() => ({}))

    await expect(saveBotMeta('researcher', { custom: true, shape: 'circle' })).resolves.toEqual({
      serverOutcome: 'unsupported',
      serverPersisted: false
    })
  })

  it('treats a rejected configure the same way', async () => {
    hostMock.request.mockRejectedValue(new Error('param shape rejected'))

    await expect(saveBotMeta('researcher', { custom: true, shape: 'circle' })).resolves.toMatchObject({
      serverOutcome: 'unsupported'
    })
  })
})

describe('a save for a bot on the pooled local backend', () => {
  it('is admitted while warm bot backends hold every background slot', async () => {
    // Main's pool admission, reduced: cap 3 with one slot reserved for
    // foreground dials. Two warm bots hold both background slots and beta's
    // backend is cold, so a dial without an explicit priority (background,
    // main's default) waits out its slot deadline and fails.
    hostMock.requestProfile.mockImplementation(
      async (
        _route: unknown,
        method: string,
        _params: unknown,
        _timeoutMs?: number,
        options?: { spawnPriority?: 'background' | 'foreground' }
      ) => {
        if ((options?.spawnPriority ?? 'background') === 'background') {
          throw new Error('Local backend start for "beta" timed out while waiting for a free slot. (background)')
        }

        return method === 'profiles.configure'
          ? { applied: { ui_meta: true } }
          : method === 'profiles.list'
            ? SERVED_PROFILES
            : {}
      }
    )

    const beta = { connectionId: 'local', connectionKind: 'local', name: 'beta', sourceScoped: true } as RosterRow

    await expect(saveBotMeta(beta, { sectionId: 'sec-1', sectionName: 'Clients' })).resolves.toEqual({
      serverOutcome: 'persisted',
      serverPersisted: true
    })
  })
})

describe('avatar asset sync fires only on a real change', () => {
  // Edit Profile always sends the image key, changed or not. Firing
  // set_asset for every patch re-uploaded the full data URL — and a no-op
  // `clear` from one machine could race another machine's just-pushed
  // avatar and wipe it server-side.
  it('pushes once, skips the re-save, clears on removal, ignores an absent key', async () => {
    const calls = recordRequests()
    const png = 'data:image/png;base64,AAAA'

    await saveBotMeta('ops', { image: png, title: 'One' })
    await saveBotMeta('ops', { image: png, title: 'Two' })
    await saveBotMeta('ops', { image: null, title: 'Three' })
    await saveBotMeta('ops', { title: 'Four' })

    expect(calls.filter(call => call.method === 'profiles.set_asset').map(call => call.params)).toEqual([
      { asset: 'avatar', data: png, name: 'ops' },
      { asset: 'avatar', clear: true, name: 'ops' }
    ])
    // Meta still merges per patch, and ui_meta sync is unaffected.
    expect($botMeta.get().ops.title).toBe('Four')
    expect(calls.filter(call => call.method === 'profiles.configure')).toHaveLength(4)
  })

  it('still pushes the copied avatar when a bot is duplicated', async () => {
    const calls = recordRequests()
    const png = 'data:image/png;base64,BBBB'

    await saveBotMeta('source', { image: png, title: 'Original' })
    await saveBotMeta('source-2', { image: png, title: 'Original (copy)' })

    // The fresh profile's image differs from its (empty) meta, so it pushes.
    expect(calls.filter(call => call.method === 'profiles.set_asset').map(call => call.params)).toEqual([
      { asset: 'avatar', data: png, name: 'source' },
      { asset: 'avatar', data: png, name: 'source-2' }
    ])
  })

  it('never ships the image or the pet icon inside ui_meta', async () => {
    // ui_meta rides every profiles.list and is capped at 64KB; the image goes
    // to the uncapped profile asset store instead.
    const calls = recordRequests()

    await saveBotMeta('ops', { image: 'data:image/png;base64,CCCC', pet: 'pet-cat', title: 'Ops' })

    const configure = calls.find(call => call.method === 'profiles.configure')
    const uiMeta = (configure?.params.ui_meta as Record<string, Record<string, unknown>>)['hermes-bots']

    expect(uiMeta).toEqual({ title: 'Ops' })
  })
})

describe('the disk load never clobbers a write from this session', () => {
  async function loadWithStorage(value: unknown) {
    vi.resetModules()

    const data = await import('./data')

    storageMock.get.mockImplementation(async (key: string) => (key === 'bot-meta' ? await value : null))

    return data
  }

  it('keeps a value written while the load was still in flight', async () => {
    let resolveStorage!: (value: unknown) => void

    const pending = new Promise(resolve => {
      resolveStorage = resolve
    })

    const data = await loadWithStorage(pending)

    void data.saveBotMeta('researcher', { pinned: true, title: 'Research' })
    expect(data.$botMeta.get().researcher.pinned).toBe(true)

    const migration = data.migrateBotMeta(storageMock)

    resolveStorage({ researcher: { title: 'Research' } })
    await migration

    expect(data.$botMeta.get().researcher).toMatchObject({ pinned: true, title: 'Research' })
  })

  it('still merges looks from disk under the fresher in-session write', async () => {
    const data = await loadWithStorage({ researcher: { color: '#38bdf8', shape: 'cloud', title: 'Research' } })

    void data.saveBotMeta('researcher', { pinned: true })
    await data.migrateBotMeta(storageMock)

    expect(data.$botMeta.get().researcher).toMatchObject({
      color: '#38bdf8',
      pinned: true,
      shape: 'cloud',
      title: 'Research'
    })
  })

  it('fills in bots this session has not touched', async () => {
    const data = await loadWithStorage({
      researcher: { pinned: true, title: 'Research' },
      writer: { title: 'Writer' }
    })

    await data.migrateBotMeta(storageMock)

    expect(data.$botMeta.get().researcher.pinned).toBe(true)
    expect(data.$botMeta.get().writer.title).toBe('Writer')
  })
})

describe('server metadata reconciliation', () => {
  const serverRow = (name: string, meta: Record<string, unknown>) =>
    ({ name, ui_meta: { 'hermes-bots': meta } }) as unknown as RosterRow

  it('removes a stale local canonical-chat pointer on sight', () => {
    // Identity is the profile's "Bot Chat" registry row, resolved by name.
    // Old `meta.chat` pointers must never look meaningful again.
    $botMeta.set({
      default: { chat: '4f6f6798ad6f', image: 'data:image/png;base64,avatar', pet: 'pet-cat', shape: 'cloud' }
    })

    mergeServerMeta([serverRow('default', { color: '#8b5cf6', shape: 'cloud', title: 'Hermana' })])

    const current = $botMeta.get().default

    expect(Object.hasOwn(current, 'chat')).toBe(false)
    // Local-only fields the server copy never carries survive the overlay.
    expect(current.image).toBe('data:image/png;base64,avatar')
    expect(current.pet).toBe('pet-cat')

    const [key, value] = storageMock.set.mock.calls.at(-1) as [string, Record<string, Record<string, unknown>>]

    expect(key).toBe('bot-meta')
    expect(Object.hasOwn(value.default, 'chat')).toBe(false)
  })

  it('drops the legacy scalar when authoritative groups say the membership is gone', () => {
    // A server-side `group: null` is represented by OMISSION, so retaining
    // the local scalar resurrects a membership another desktop just removed.
    $botMeta.set({ researcher: { group: 'Old', groups: ['Old'], title: 'Researcher' } })

    mergeServerMeta([serverRow('researcher', { groups: [], title: 'Researcher' })])

    const current = $botMeta.get().researcher

    expect(current.groups).toEqual([])
    expect(Object.hasOwn(current, 'group')).toBe(false)
  })

  it('leaves local metadata alone when the gateway has none', () => {
    $botMeta.set({ default: { chat: 'local-chat', shape: 'cloud' } })

    mergeServerMeta([{ name: 'default' } as RosterRow])

    expect($botMeta.get().default.chat).toBe('local-chat')
    expect(storageMock.set).not.toHaveBeenCalled()
  })

  it('cannot resurrect a membership a later local write removed (disband-resurrection)', async () => {
    $botMeta.set({ builder: { group: 'Team', groups: ['Team'], title: 'Builder' } })

    // Snapshot fetched now — its ui_meta still names the group.
    const staleFetchedAt = Date.now()
    const staleRow = serverRow('builder', { group: 'Team', groups: ['Team'], title: 'Builder' })

    // Disband path: the local write lands AFTER that fetch.
    await new Promise(resolve => setTimeout(resolve, 5))
    await saveBotMeta('builder', { group: null, groups: [] })
    expect($botMeta.get().builder.groups).toEqual([])

    mergeServerMeta([staleRow], staleFetchedAt)
    expect($botMeta.get().builder.groups).toEqual([])

    // A FRESH snapshot still gets the last word — server truth wins once it
    // actually post-dates the local change.
    mergeServerMeta([staleRow], Date.now() + 5)
    expect($botMeta.get().builder.groups).toEqual(['Team'])
  })

  it('overlays without a fence when the caller has no fetch time (older callers)', async () => {
    await saveBotMeta('builder', { group: null, groups: [] })
    mergeServerMeta([serverRow('builder', { groups: ['Team'] })])

    expect($botMeta.get().builder.groups).toEqual(['Team'])
  })

  it('forgets write stamps far too old to fence any snapshot', async () => {
    vi.useFakeTimers()

    try {
      await saveBotMeta('builder', { title: 'Builder' })

      expect([...botMetaWriteAt.keys()]).toEqual(['builder'])

      // The renderer stays up for days. Once no snapshot can still be in
      // flight from before a stamp, keeping it just costs a row per bot.
      vi.advanceTimersByTime(120_000)
      await saveBotMeta('researcher', { title: 'Research' })

      expect([...botMetaWriteAt.keys()]).toEqual(['researcher'])
    } finally {
      vi.useRealTimers()
    }
  })
})

// Two machines, both with a profile literally named `default`: the VPS bot is
// "Agent A" on its own backend; this Desktop's local agent is "Agent B", whose
// title still sits under the bare legacy key from its single-source days.
describe("a bot is named by its own backend, never by another bot's cached record", () => {
  const vps = {
    connectionId: 'vps',
    display_name: 'Agent A',
    name: 'default',
    remoteSource: true,
    sourceScoped: true
  } as RosterRow

  it('a save writes its own patch onto the server namespace, and re-reads after losing a CAS race', async () => {
    $botMeta.set({ 'vps::default': { color: '#ff0000', title: 'Agent B' } })
    let revision = 3
    const configures: Array<Record<string, unknown>> = []

    hostMock.requestProfile.mockImplementation(
      async (_route: unknown, method: string, params: Record<string, unknown>) => {
        if (method === 'profiles.list') {
          return {
            profiles: [
              {
                name: 'default',
                ui_meta: { 'hermes-bots': { color: '#0000ff', groups: [] } },
                ui_meta_revisions: { 'hermes-bots': revision }
              }
            ]
          }
        }

        if (method === 'profiles.configure') {
          configures.push(structuredClone(params))

          // Another writer lands between this save's read and its write, once.
          if (configures.length === 1) {
            revision = 4

            return { applied: { ui_meta: false, ui_meta_conflicts: { 'hermes-bots': { actual: 4, expected: 3 } } } }
          }

          return { applied: { ui_meta: true } }
        }

        return {}
      }
    )

    await expect(saveBotMeta(vps, { groups: ['room'] })).resolves.toMatchObject({ serverOutcome: 'persisted' })

    expect(configures.map(params => params.ui_meta_expected_revisions)).toEqual([
      { 'hermes-bots': 3 },
      { 'hermes-bots': 4 }
    ])
    // The local cache's stale title and color never travel: only the patch rides the server's copy.
    expect(configures.at(-1)?.ui_meta).toEqual({ 'hermes-bots': { color: '#0000ff', groups: ['room'] } })
  })

  it('names, tags, and prompts the bot with what its backend reports', () => {
    $botMeta.set({ default: { title: 'Agent B' }, 'vps::default': { title: 'Agent B' } })
    // `/api/profiles` reports no Bot Mode title for the VPS bot.
    const thin = { ...vps, title: '' } as RosterRow

    mergeServerMeta([thin])

    expect(displayName(thin, botRosterMeta(thin, $botMeta.get()))).toBe('Agent A')
    expect(botMentionTag(thin)).toBe('agent-a')

    // Window switched to the VPS: its rows are the active source's (scoped, not remote).
    const active = { ...vps, remoteSource: false, ui_meta_revisions: {} } as RosterRow

    expect(buildGroupChatTurnPrompt({ deltaLines: [], groupName: 'g', members: [active], viewer: active })).toContain(
      'You are @agent-a,'
    )

    // A keyed speaker with no live roster row (Bots pane not mounted).
    $lastRoster.set([])
    expect(groupSpeakerLabel('vps::default')).not.toBe('Agent B')
  })

  it("only a fresh answer from the bot's backend corrects its record, and a title alone never mints one", () => {
    $botMeta.set({ 'vps::default': { color: '#00ff00', title: 'New title' } })
    // The VPS stops answering; the pane keeps its last-painted row, which predates the rename.
    const painted = { ...vps, title: 'Old title' } as RosterRow

    const kept = mergeMultiSourceRoster(
      { profiles: [] },
      { agents: [], sources: [{ connectionId: 'vps', reachable: false }] },
      'local',
      [painted]
    )

    mergeServerMeta(kept.profiles as RosterRow[], Date.now())
    expect($botMeta.get()['vps::default']).toEqual({ color: '#00ff00', title: 'New title' })

    // No record yet (a configured alias may own this bot's look): the backend's
    // title rides the row, and no empty record appears to shadow the alias.
    $botMeta.set({})
    mergeServerMeta([{ ...vps, title: 'Agent A' } as RosterRow])
    expect($botMeta.get()).toEqual({})
  })

  it('Edit Profile sends only what the user changed since it opened', () => {
    const opened = { color: '#ff0000', image: null, shape: 'circle' as const, title: 'Old title' }

    // Renamed elsewhere meanwhile; this dialog only changes the color.
    expect(editedLook(opened, { ...opened, color: '#0000ff' })).toEqual({ color: '#0000ff', custom: true })
    expect(editedLook(opened, { ...opened, title: ' Old title ' })).toBeNull()
  })
})

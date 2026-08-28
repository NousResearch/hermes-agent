import { beforeEach, describe, expect, it, vi } from 'vitest'

import type * as groupChat from './group-chat'
import type * as groupRounds from './group-rounds'
import { createGroupGateway, deferTimers, drain, runTimersInline, scriptedStorage } from './group-test-utils'
import type { GatewayOptions, ScriptedGateway } from './group-test-utils'
import type { GroupChat, GroupMessage } from './types'

// The room store: the atom every group surface reads, the durable projection
// written to plugin storage, and the bounded mirror published to the gateway's
// profile ui_meta so mobile sees the same rooms. Identity is the load-bearing
// part — a room is a roomId, not a display name, and every merge has to agree
// about that across devices that were offline for different windows.

const { host } = vi.hoisted(() => ({ host: {} as Record<string, unknown> }))

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')

  return pluginSdkMock(host)
})

interface Room {
  chat: typeof groupChat
  gateway: ScriptedGateway
  rounds: typeof groupRounds
}

/** The sync envelope, borrowed from the merge entry point — the interface
 *  itself is private to group-chat.ts. */
type SyncSnapshot = Parameters<typeof groupChat.mergeGroupChatSyncSnapshots>[0]

async function loadRoom(options: GatewayOptions = {}): Promise<Room> {
  vi.resetModules()
  const gateway = createGroupGateway(options)

  for (const key of Object.keys(host)) {
    delete host[key]
  }

  Object.assign(host, gateway.host)

  const [chat, rounds, shared] = await Promise.all([
    import('./group-chat'),
    import('./group-rounds'),
    import('./shared')
  ])

  shared.setPluginCtx(scriptedStorage(gateway.storage))

  return { chat, gateway, rounds }
}

const durable = (room: Room) => (room.gateway.storage.get('group-chats') || {}) as Record<string, GroupChat>

/** The last mirror the sync worker published to the default profile. */
function published(room: Room) {
  const configure = room.gateway.rpcFor('profiles.configure').at(-1)

  return (configure?.params.ui_meta as Record<string, Record<string, unknown>>)?.['hermes-bots-groups']
}

beforeEach(() => {
  runTimersInline()
})

describe('log window', () => {
  it('trimming keeps watermarks consistent with the trimmed array', async () => {
    const { chat } = await loadRoom()

    const log = Array.from({ length: 200 }, (_, i) => ({
      at: i,
      from: { kind: 'user', name: 'You' },
      text: `m${i}`
    })) as GroupMessage[]

    const trimmed = chat.trimGroupChatLog(log, { builder: 10, research: 150 }, 96)

    expect(trimmed.log).toHaveLength(96)
    expect(trimmed.watermarks.research).toBe(150 - 104)
    expect(trimmed.watermarks.builder).toBe(0)
  })
})

describe('room naming', () => {
  it('same-name dedup reserves suffix length at the 64-char cap', async () => {
    const { chat } = await loadRoom()

    expect(chat.uniqueGroupChatName('Team', new Set(['Other']))).toBe('Team')

    // Slicing the joined string would chop the " 2"/" 3" suffix off a
    // max-length base and collide with the original forever.
    const base = 'x'.repeat(64)
    const next = chat.uniqueGroupChatName(base, new Set([base, `${base.slice(0, 62)} 2`]))

    expect(next).not.toBe(base)
    expect(next).toHaveLength(64)
    expect(next).toBe(`${base.slice(0, 62)} 3`)
  })
})

describe('speaker labels', () => {
  it('the default profile speaks as Hermes in transcripts, not @default', async () => {
    const { rounds } = await loadRoom()

    const line = rounds.formatGroupChatLine(
      { from: { kind: 'member', name: 'default' }, text: 'hello room' } as GroupMessage,
      'builder'
    )

    expect(line).toBe('Hermes: hello room')

    // Other members keep their profile name; the (you) suffix survives.
    expect(
      rounds.formatGroupChatLine({ from: { kind: 'member', name: 'default' }, text: 'hi' } as GroupMessage, 'default')
    ).toBe('Hermes (you): hi')
    expect(
      rounds.formatGroupChatLine({ from: { kind: 'member', name: 'builder' }, text: 'yo' } as GroupMessage, 'research')
    ).toBe('builder: yo')
  })

  it('honor friendly identity: Bot Mode title, then display_name, never a stale Hermes', async () => {
    const { chat, rounds } = await loadRoom()
    const data = await import('./data')

    // A renamed default (core display_name via `hermes profile rename`) must
    // read as its new name — the community report was "Lucy" still showing
    // "Hermes is thinking…" in group rooms.
    data.$lastRoster.set([{ display_name: 'Lucy', name: 'default' }])

    expect(chat.groupSpeakerLabel('default')).toBe('Lucy')
    expect(
      rounds.formatGroupChatLine({ from: { kind: 'member', name: 'default' }, text: 'hi' } as GroupMessage, 'builder')
    ).toBe('Lucy: hi')

    // A Bot Mode title outranks display_name (same precedence as displayName).
    data.$botMeta.set({ default: { title: 'Moxie' } })

    expect(chat.groupSpeakerLabel('default')).toBe('Moxie')

    // Secondary profiles get their title too — the thinking line names the
    // renamed bot, not the raw profile slug.
    data.$botMeta.set({ research: { title: 'Radar' } })
    data.$lastRoster.set([])

    expect(chat.groupSpeakerLabel('research')).toBe('Radar')

    // Untitled rows keep today's behavior: default → Hermes, others verbatim.
    data.$botMeta.set({})

    expect(chat.groupSpeakerLabel('default')).toBe('Hermes')
    expect(chat.groupSpeakerLabel('builder')).toBe('builder')
  })

  it("never borrow a remote row's display_name for a local speaker", async () => {
    const { chat } = await loadRoom()
    const data = await import('./data')
    // Only a remote/thin row named default exists — its display_name belongs
    // to that connection, not to the active gateway's default.
    data.$lastRoster.set([{ display_name: 'HomelabBot', name: 'default', remoteSource: true }])

    expect(chat.groupSpeakerLabel('default')).toBe('Hermes')
  })
})

// #93127: duplicate room delivery. Two raceable paths existed: a member turn
// mid-flight when the room epoch bumps still committed its reply + watermark
// (the stale loop only noticed supersession at the NEXT member boundary), and
// the stale loop and the fresh loop could both append the same reply.
describe('turn commit gate (#93127)', () => {
  it('discards a superseded turn — the epoch moved on while the turn ran', async () => {
    const { chat } = await loadRoom()

    expect(chat.shouldCommitMemberTurn(3, 4)).toBe(false)
    expect(chat.shouldCommitMemberTurn(3, 7)).toBe(false)
    // Explicit same-thread supersession.
    expect(chat.shouldCommitMemberTurn(3, 4, true)).toBe(false)
  })

  it('commits a current turn — the epoch is unchanged since dispatch', async () => {
    const { chat } = await loadRoom()

    expect(chat.shouldCommitMemberTurn(3, 3)).toBe(true)
    expect(chat.shouldCommitMemberTurn(0, 0)).toBe(true)
  })

  it('does NOT discard finished work on a cross-thread epoch bump', async () => {
    const { chat } = await loadRoom()

    // Epoch moved, but no newer user entry landed in THIS thread: the
    // superseding send lives in another thread whose loop filters this one
    // out — dropping the reply would lose completed work forever.
    expect(chat.shouldCommitMemberTurn(3, 4, false)).toBe(true)
  })
})

describe('duplicate append guard (#93127)', () => {
  const seed = (chat: typeof groupChat, entry: GroupMessage) => {
    chat.$groupChats.set({ Room: { log: [entry], watermarks: {} } })
  }

  const lastEntry = (name: string, text: string, thread = 't1', at = Date.now(), source?: string): GroupMessage =>
    ({
      at,
      from: { kind: 'member', name, ...(source ? { source } : {}) },
      id: 'x',
      text,
      thread
    }) as GroupMessage

  it('drops an adjacent identical member reply', async () => {
    const { chat } = await loadRoom()
    seed(chat, lastEntry('impl', 'Stopped. Standing by.'))

    chat.appendGroupChatEntry('Room', { kind: 'member', name: 'impl' }, 'Stopped. Standing by.', 't1')

    expect(chat.$groupChats.get().Room.log).toHaveLength(1)
  })

  it('keeps identical text from a DIFFERENT member', async () => {
    const { chat } = await loadRoom()
    seed(chat, lastEntry('impl', 'Confirmed.'))

    chat.appendGroupChatEntry('Room', { kind: 'member', name: 'reviewer' }, 'Confirmed.', 't1')

    expect(chat.$groupChats.get().Room.log).toHaveLength(2)
  })

  it('keeps identical text from the same member on another SOURCE', async () => {
    const { chat } = await loadRoom()
    seed(chat, lastEntry('impl', 'Confirmed.', 't1', Date.now(), 'laptop'))

    chat.appendGroupChatEntry('Room', { kind: 'member', name: 'impl' }, 'Confirmed.', 't1')

    expect(chat.$groupChats.get().Room.log).toHaveLength(2)
  })

  it('keeps the same text after an intervening entry — only the LAST entry is checked', async () => {
    const { chat } = await loadRoom()
    seed(chat, lastEntry('reviewer', 'ack'))

    chat.appendGroupChatEntry('Room', { kind: 'member', name: 'impl' }, 'Confirmed.', 't1')

    expect(chat.$groupChats.get().Room.log).toHaveLength(2)
  })

  it('keeps identical text in a DIFFERENT thread', async () => {
    const { chat } = await loadRoom()
    seed(chat, lastEntry('impl', 'Confirmed.', 'thread-a'))

    chat.appendGroupChatEntry('Room', { kind: 'member', name: 'impl' }, 'Confirmed.', 'thread-b')

    expect(chat.$groupChats.get().Room.log).toHaveLength(2)
  })

  it('keeps identical text outside the recency window', async () => {
    const { chat } = await loadRoom()
    seed(chat, lastEntry('impl', 'Done.', 't1', Date.now() - 60 * 60 * 1000))

    chat.appendGroupChatEntry('Room', { kind: 'member', name: 'impl' }, 'Done.', 't1')

    expect(chat.$groupChats.get().Room.log).toHaveLength(2)
  })

  it('never dedupes user entries', async () => {
    const { chat } = await loadRoom()
    seed(chat, { at: Date.now(), from: { kind: 'user', name: 'You' }, id: 'x', text: 'stop', thread: 't1' })

    chat.appendGroupChatEntry('Room', { kind: 'user', name: 'You' }, 'stop', 't1')

    expect(chat.$groupChats.get().Room.log).toHaveLength(2)
  })
})

describe('threads', () => {
  it('hydration assigns legacy thread ids — a lull splits, follow-ups stay together', async () => {
    const { chat } = await loadRoom()
    const minute = 60000
    const user = (text: string, at: number) => ({ at, from: { kind: 'user', name: 'You' }, text }) as GroupMessage

    const member = (name: string, text: string, at: number) =>
      ({ at, from: { kind: 'member', name }, text }) as GroupMessage

    const log = chat.assignLegacyThreads([
      user('task one', 0),
      member('a', 'r1', 1 * minute),
      user('quick follow-up', 3 * minute), // inside the 15-min window: SAME thread
      member('a', 'r2', 4 * minute),
      user('new topic much later', 60 * minute), // after the lull: new thread
      member('a', 'r3', 61 * minute)
    ])

    expect(log[0].thread).toBe(log[2].thread)
    expect(log[2].thread).toBe(log[3].thread)
    expect(log[0].thread).not.toBe(log[4].thread)
    expect(log[4].thread).toBe(log[5].thread)
  })
})

describe('durable projection', () => {
  it('excludes tombstoned rooms', async () => {
    const { chat } = await loadRoom()

    const rooms = chat.durableGroupChatRooms({
      Keep: { log: [{ at: 1, from: { kind: 'user' }, text: 'hi' }], members: [], watermarks: {} },
      Live: { epoch: 4, log: [], running: false, tombstone: true, watermarks: {} }
    } as unknown as Record<string, GroupChat>)

    expect('Live' in rooms).toBe(false)
    expect('Keep' in rooms).toBe(true)
  })

  it('carries roomId — omitting it would drop the durable id on the next cold hydrate', async () => {
    const { chat } = await loadRoom()

    const rooms = chat.durableGroupChatRooms({
      Legacy: { log: [{ at: 1, from: { kind: 'user' }, text: 'hi' }], members: [], watermarks: {} },
      Team: { log: [{ at: 1, from: { kind: 'user' }, text: 'hi' }], members: [], roomId: 'room-abc123', watermarks: {} }
    } as unknown as Record<string, GroupChat>)

    expect(rooms.Team.roomId).toBe('room-abc123')
    // An explicit null, not undefined — the key has to survive JSON.
    expect(rooms.Legacy.roomId).toBeNull()
  })

  it('never persists a tombstone that a remote merge forwarded', async () => {
    const room = await loadRoom()
    // A drive still mid-turn at disband time leaves a live tombstone.
    room.chat.$groupChats.set({
      Live: { epoch: 4, log: [], running: false, tombstone: true, watermarks: {} }
    } as unknown as Record<string, GroupChat>)

    // The remote gateway has NOT yet received the delete (plausible now that
    // sync fans out to every reachable default-profile gateway independently)
    // — its snapshot still carries a live copy under the same display name.
    const merged = room.chat.mergeRemoteGroupChatSnapshotIntoRooms(
      {
        rooms: {
          Live: {
            log: [{ at: 1, from: { kind: 'member', name: 'research' }, text: 'still going' }],
            members: [{ name: 'research' }]
          }
        },
        version: 3
      },
      room.chat.$groupChats.get()
    )

    // The merge spreads `...existing` before its explicit field overrides,
    // none of which touch `tombstone` — so the flag survives into the merged
    // room. Without that reachability step durableGroupChatRooms would never
    // see a tombstoned room from this path at all.
    expect(merged.Live.tombstone).toBe(true)

    await room.chat.persistGroupChatRooms(merged)

    expect('Live' in durable(room)).toBe(false)
  })

  it('stranded markers ride the durable map so late replies survive a reload', async () => {
    const room = await loadRoom()

    room.chat.updateGroupChat('Persist', current => {
      current.stranded = { research: 3 }

      return current
    })

    expect(durable(room).Persist.stranded?.research).toBe(3)
  })
})

describe('gateway mirror', () => {
  it('mirrors room messages and members through bounded profile metadata', async () => {
    const room = await loadRoom()

    room.rounds.sendToGroupChat(
      'Research',
      [
        { handle: 'research', name: 'research' },
        { handle: 'builder', name: 'builder' }
      ],
      'What changed?'
    )
    await drain(() => Boolean(room.chat.$groupChats.get().Research?.running))

    const envelope = published(room)
    const rooms = envelope.rooms as Record<string, Record<string, unknown>>
    const key = Object.keys(rooms).find(name => rooms[name].name === 'Research')

    expect(envelope.version).toBe(3)
    expect(key).toBeDefined()
    expect((rooms[key!].log as GroupMessage[])[0].text).toBe('What changed?')
    expect((rooms[key!].members as { name: string }[]).map(member => member.name)).toEqual(['research', 'builder'])
    expect(room.chat.groupChatGatewayJsonSize(envelope)).toBeLessThanOrEqual(48000)
  })

  it('is size bounded and favors recent messages', async () => {
    const { chat } = await loadRoom()
    const long = 'x'.repeat(5000)

    const snapshot = chat.groupChatSyncSnapshot({
      Large: {
        log: Array.from({ length: 100 }, (_, index) => ({
          at: index,
          from: { kind: index % 2 ? 'member' : 'user', name: index % 2 ? 'research' : 'You' },
          text: `${index}:${long}`
        }))
      }
    } as unknown as Record<string, GroupChat>)

    const log = snapshot.rooms['name:Large'].log

    expect(chat.groupChatGatewayJsonSize(snapshot)).toBeLessThanOrEqual(48000)
    expect(log.length).toBeLessThanOrEqual(16)
    expect(log.at(-1)?.text).toMatch(/^99:/)
    expect(log.at(-1)?.text.length).toBeLessThanOrEqual(1200)
  })

  it('preserves threads and budgets escaped Unicode', async () => {
    const { chat } = await loadRoom()

    const snapshot = chat.groupChatSyncSnapshot({
      Unicode: {
        log: Array.from({ length: 16 }, (_, index) => ({
          at: index,
          from: { kind: 'member', name: 'research' },
          text: `message ${index} ${'🧠'.repeat(1200)}`,
          thread: `thread-${index}`
        }))
      }
    } as unknown as Record<string, GroupChat>)

    expect(chat.groupChatGatewayJsonSize(snapshot)).toBeLessThanOrEqual(48000)
    expect(snapshot.rooms['name:Unicode'].log.at(-1)?.thread).toBe('thread-15')
  })

  it('omits empty runtime rooms', async () => {
    const { chat } = await loadRoom()

    const snapshot = chat.groupChatSyncSnapshot({
      Disbanded: { log: [], members: [{ name: 'research' }] }
    } as unknown as Record<string, GroupChat>)

    expect(Object.keys(snapshot.rooms)).toEqual([])
  })

  it('an empty hydrate cannot erase a shared room mirror', async () => {
    const room = await loadRoom()
    const before = room.gateway.rpc.length

    room.chat.scheduleGroupChatServerSync({})

    expect(room.gateway.rpc).toHaveLength(before)
  })

  it('an explicit final-room disband may clear the mirror', async () => {
    const room = await loadRoom()

    room.chat.scheduleGroupChatServerSync({}, { allowEmpty: true, deletedRooms: ['Research'] })
    await drain(() => room.gateway.rpcFor('profiles.configure').length < 1, 50)

    const envelope = published(room)

    expect(Object.keys(envelope.rooms as object)).toEqual([])
    expect((envelope.deleted as Record<string, number>)['name:Research']).toBeGreaterThan(0)
  })
})

describe('snapshot merge', () => {
  it('pull-before-push preserves disjoint rooms, messages and members', async () => {
    const { chat } = await loadRoom()

    const merged = chat.mergeGroupChatSyncSnapshots(
      {
        rooms: {
          RemoteOnly: {
            log: [{ at: 2, from: { kind: 'member', name: 'ops' }, text: 'kept' }],
            members: [{ name: 'ops' }]
          },
          Shared: {
            log: [{ at: 1, from: { kind: 'user', name: 'You' }, text: 'remote', thread: 'one' }],
            members: [{ handle: 'research', name: 'research' }]
          }
        },
        version: 1
      },
      {
        rooms: {
          Shared: {
            log: [{ at: 3, from: { kind: 'member', name: 'builder' }, text: 'local', thread: 'one' }],
            members: [{ handle: 'builder', name: 'builder' }]
          }
        },
        version: 1
      }
    )

    expect(merged.rooms['name:Shared'].log.map(entry => entry.text)).toEqual(['remote', 'local'])
    expect(merged.rooms['name:Shared'].members?.map(member => member.name)).toEqual(['research', 'builder'])
    expect(merged.rooms['name:RemoteOnly'].log[0].text).toBe('kept')
  })

  it('a deletion tombstone wins over stale history but not a later recreation', async () => {
    const { chat } = await loadRoom()

    const stale = chat.mergeGroupChatSyncSnapshots(
      {
        rooms: {
          Research: { log: [{ at: 10, from: { kind: 'user', name: 'You' }, id: 'old', text: 'old' }], revision: 1 }
        },
        version: 2
      },
      { deleted: { Research: 2 }, rooms: {}, version: 2 }
    )

    expect(stale.rooms['name:Research']).toBeUndefined()
    expect(stale.deleted?.['name:Research']).toBe(2)

    const recreated = chat.mergeGroupChatSyncSnapshots(stale, {
      rooms: {
        Research: { log: [{ at: 1, from: { kind: 'user', name: 'You' }, id: 'new', text: 'new' }], revision: 3 }
      },
      version: 2
    })

    expect(recreated.rooms['name:Research'].log[0].text).toBe('new')
    expect(recreated.deleted).toBeUndefined()
  })

  it('orders deletion and recreation by gateway revision, not device clocks', async () => {
    const { chat } = await loadRoom()

    const merged = chat.mergeGroupChatSyncSnapshots(
      {
        rooms: {
          ClockSkewed: {
            log: [{ at: 9999999999999, from: { kind: 'user', name: 'You' }, id: 'future-clock', text: 'old' }],
            revision: 8
          }
        },
        version: 2
      },
      { deleted: { ClockSkewed: 9 }, rooms: {}, version: 2 }
    )

    expect(merged.rooms['name:ClockSkewed']).toBeUndefined()
    expect(merged.deleted?.['name:ClockSkewed']).toBe(9)
  })

  it('lets a conflicting writer merge the winner and preserve both stable message ids', async () => {
    const { chat } = await loadRoom()

    const winner: SyncSnapshot = {
      rooms: {
        Shared: {
          log: [{ at: 100, from: { kind: 'user', name: 'You' }, id: 'writer-a:1', text: 'alpha' }],
          members: [{ name: 'alpha' }],
          revision: 1
        }
      },
      version: 2
    }

    const loserRetry = chat.mergeGroupChatSyncSnapshots(
      winner,
      {
        rooms: {
          Shared: {
            log: [{ at: 1, from: { kind: 'member', name: 'beta' }, id: 'writer-b:1', text: 'beta' }],
            members: [{ name: 'beta' }],
            revision: 0
          }
        },
        version: 2
      },
      { changedRooms: ['Shared'], writeRevision: 2 }
    )

    expect(loserRetry.rooms['name:Shared'].log.map(entry => entry.id).sort()).toEqual(['writer-a:1', 'writer-b:1'])
    expect(loserRetry.rooms['name:Shared'].revision).toBe(2)
  })
})

describe('cold hydrate', () => {
  it('projects gateway rooms without dropping local runtime fields', async () => {
    const { chat } = await loadRoom()

    const merged = chat.mergeRemoteGroupChatSnapshotIntoRooms(
      {
        rooms: {
          Shared: {
            log: [
              { at: 10, from: { kind: 'user', name: 'You' }, text: 'remote question', thread: 'thread-1' },
              { at: 20, from: { kind: 'member', name: 'research' }, text: 'remote answer', thread: 'thread-1' }
            ],
            members: [{ connectionId: 'mini', name: 'research', sourceScoped: true }]
          }
        },
        version: 3
      },
      {
        Local: {
          epoch: 7,
          log: [{ at: 5, from: { kind: 'user', name: 'You' }, text: 'local', thread: 'thread-local' }],
          members: [{ name: 'builder' }],
          running: true,
          sessions: { builder: 'session-1' },
          watermarks: { builder: 1 }
        }
      } as unknown as Record<string, GroupChat>
    )

    expect(merged.Shared.log.map(entry => entry.text)).toEqual(['remote question', 'remote answer'])
    expect(merged.Shared.members?.[0].remoteSource).toBe(true)
    expect(merged.Local.sessions?.builder).toBe('session-1')
    expect(merged.Local.epoch).toBe(7)
    expect(merged.Local.running).toBe(true)
  })

  it('read-back does not clobber a newer local edit queued during an in-flight write', async () => {
    const { chat } = await loadRoom()

    const merged = chat.mergeRemoteGroupChatSnapshotIntoRooms(
      {
        rooms: {
          Shared: {
            image: 'data:image/png;base64,old',
            log: [{ at: 1, from: { kind: 'user', name: 'You' }, id: 'remote', text: 'remote message' }],
            members: [{ name: 'old-member' }],
            revision: 5
          }
        },
        version: 2
      },
      {
        Shared: {
          image: 'data:image/png;base64,new',
          log: [{ at: 2, from: { kind: 'user', name: 'You' }, id: 'local', text: 'local message' }],
          members: [{ name: 'new-member' }],
          syncRevision: 4
        }
      } as unknown as Record<string, GroupChat>,
      { preserveRooms: ['Shared'] }
    )

    expect(merged.Shared.members?.map(member => member.name)).toEqual(['new-member'])
    expect(merged.Shared.image).toBe('data:image/png;base64,new')
    expect(merged.Shared.syncRevision).toBe(4)
    expect(merged.Shared.log.map(entry => entry.id).sort()).toEqual(['local', 'remote'])
  })
})

// A room's identity is its roomId. Display names are labels: renaming one is a
// field update on the same key, and a name-keyed tombstone must never be able
// to kill an id-keyed room (or vice versa).
describe('room identity', () => {
  it('rename publishes a new room plus an old-name tombstone that cold hydrate cannot resurrect', async () => {
    const { chat } = await loadRoom()

    const before: SyncSnapshot = {
      rooms: {
        Old: {
          image: 'data:image/png;base64,room',
          log: [{ at: 10, from: { kind: 'user', name: 'You' }, id: 'turn-1', text: 'history' }],
          members: [{ name: 'research' }],
          revision: 4
        }
      },
      version: 2
    }

    const after = chat.mergeGroupChatSyncSnapshots(
      before,
      {
        rooms: {
          New: {
            image: before.rooms.Old.image,
            log: before.rooms.Old.log,
            members: before.rooms.Old.members,
            revision: 4
          }
        },
        version: 2
      },
      { changedRooms: ['New'], deletedRooms: ['Old'], writeRevision: 5 }
    )

    const hydrated = chat.mergeRemoteGroupChatSnapshotIntoRooms(after, {})

    expect(after.rooms['name:Old']).toBeUndefined()
    expect(after.deleted?.['name:Old']).toBe(5)
    expect(after.rooms['name:New'].revision).toBe(5)
    expect(after.rooms['name:New'].image).toBe('data:image/png;base64,room')
    expect(hydrated.Old).toBeUndefined()
    expect(hydrated.New.log[0].text).toBe('history')
    expect(hydrated.New.image).toBe('data:image/png;base64,room')
  })

  it('a rename with a roomId is a same-key field update, never delete+create', async () => {
    const { chat } = await loadRoom()

    const before: SyncSnapshot = {
      rooms: {
        'id:room-42': {
          log: [{ at: 10, from: { kind: 'user', name: 'You' }, id: 'turn-1', text: 'history' }],
          members: [{ name: 'research' }],
          name: 'Old',
          revision: 4,
          roomId: 'room-42'
        }
      },
      version: 3
    }

    const after = chat.mergeGroupChatSyncSnapshots(
      before,
      { rooms: { 'id:room-42': { ...before.rooms['id:room-42'], name: 'New' } }, version: 3 },
      { changedRooms: ['id:room-42'], writeRevision: 5 }
    )

    // Same durable key, new display name, no tombstone needed at all.
    expect(Object.keys(after.rooms)).toHaveLength(1)
    expect(after.rooms['id:room-42'].name).toBe('New')
    expect(after.rooms['id:room-42'].revision).toBe(5)
    expect(after.deleted).toBeUndefined()

    // A lagging gateway still holding the OLD name under the same id cannot
    // resurrect it: same key, lower revision, identity follows the winner.
    expect(chat.mergeGroupChatSyncSnapshots(before, after).rooms['id:room-42'].name).toBe('New')
  })

  it('id tombstones are final even against a higher-revision lagging copy', async () => {
    const { chat } = await loadRoom()

    const merged = chat.mergeGroupChatSyncSnapshots(
      {
        // A gateway that was OFFLINE during the disband still carries the room
        // with a high revision from busy pre-disband traffic.
        rooms: {
          'id:room-9': {
            log: [{ at: 1, from: { kind: 'user', name: 'You' }, id: 'stale', text: 'stale' }],
            name: 'Zombie',
            revision: 40,
            roomId: 'room-9'
          }
        },
        version: 3
      },
      { deleted: { 'id:room-9': 3 }, rooms: {}, version: 3 }
    )

    expect(merged.rooms['id:room-9']).toBeUndefined()
    expect(merged.deleted?.['id:room-9']).toBe(3)

    // Same-name RECREATION is unaffected: the new room minted a fresh id.
    const recreated = chat.mergeGroupChatSyncSnapshots(merged, {
      rooms: {
        'id:room-10': {
          log: [{ at: 2, from: { kind: 'user', name: 'You' }, id: 'fresh', text: 'fresh' }],
          name: 'Zombie',
          revision: 1,
          roomId: 'room-10'
        }
      },
      version: 3
    })

    expect(recreated.rooms['id:room-10'].log[0].text).toBe('fresh')
    expect(recreated.rooms['id:room-9']).toBeUndefined()
  })

  it('cold hydrate follows a remote rename via roomId instead of duplicating', async () => {
    const { chat } = await loadRoom()

    const merged = chat.mergeRemoteGroupChatSnapshotIntoRooms(
      {
        rooms: {
          'id:room-7': {
            log: [{ at: 1, from: { kind: 'user', name: 'You' }, id: 'm1', text: 'hello' }],
            members: [{ name: 'research' }],
            name: 'Renamed',
            revision: 6,
            roomId: 'room-7'
          }
        },
        version: 3
      },
      {
        // Local copy still under the pre-rename display name, same roomId.
        Original: {
          log: [{ at: 1, from: { kind: 'user', name: 'You' }, id: 'm1', text: 'hello' }],
          members: [{ name: 'research' }],
          roomId: 'room-7',
          sessions: { research: 'sid-1' },
          syncRevision: 5,
          watermarks: {}
        }
      } as unknown as Record<string, GroupChat>
    )

    expect(merged.Original).toBeUndefined()
    expect(merged.Renamed.roomId).toBe('room-7')
    expect(merged.Renamed.sessions?.research).toBe('sid-1')
  })

  it('renaming an id-keyed room via the rename job shape never tombstones it', async () => {
    const { chat } = await loadRoom()

    const remote: SyncSnapshot = {
      rooms: {
        'id:room-3': {
          log: [{ at: 5, from: { kind: 'user', name: 'You' }, id: 'h1', text: 'history' }],
          members: [{ name: 'research' }],
          name: 'Old',
          revision: 2,
          roomId: 'room-3'
        }
      },
      version: 3
    }

    // Exactly what renameGroupChat schedules: changed [new], deleted [old].
    const after = chat.mergeGroupChatSyncSnapshots(
      remote,
      { rooms: { 'id:room-3': { ...remote.rooms['id:room-3'], name: 'New' } }, version: 3 },
      { changedRooms: ['New'], deletedRooms: ['Old'], writeRevision: 3 }
    )

    expect(after.rooms['id:room-3']).toBeDefined()
    expect(after.rooms['id:room-3'].name).toBe('New')
    expect(after.deleted?.['id:room-3']).toBeUndefined()
    // A residual name:Old tombstone is correct — it retires stale v2/legacy
    // copies of this room that older clients published under the name key.
    expect(after.deleted?.['name:Old']).toBe(3)

    // Hydrate path: the pull that races the rename write must not delete the
    // locally re-keyed record just because the remote copy still says Old.
    const merged = chat.mergeRemoteGroupChatSnapshotIntoRooms(
      remote,
      {
        New: {
          log: remote.rooms['id:room-3'].log,
          members: [{ name: 'research' }],
          roomId: 'room-3',
          sessions: { research: 'sid-9' },
          syncRevision: 2,
          watermarks: {}
        }
      } as unknown as Record<string, GroupChat>,
      { deletedRooms: ['Old'], preserveRooms: ['New'] }
    )

    expect(merged.New).toBeDefined()
    expect(merged.New.sessions?.research).toBe('sid-9')
    expect(merged.Old).toBeUndefined()
  })
})

describe('sync worker', () => {
  it('retries a gateway CAS conflict and publishes the merged room', async () => {
    deferTimers()

    // A concurrent writer lands between our read and our write: the gateway
    // rejects the CAS, and the retry has to merge rather than overwrite.
    const room = await loadRoom({
      conflictOnce: {
        key: 'hermes-bots-groups',
        value: {
          rooms: {
            'name:Shared': {
              log: [{ at: 1, from: { kind: 'user', name: 'You' }, id: 'writer-a:1', text: 'alpha' }],
              members: [{ name: 'alpha' }],
              revision: 1
            }
          },
          version: 3
        }
      }
    })

    room.chat.$groupChats.set({
      Shared: {
        log: [{ at: 2, from: { kind: 'member', name: 'beta' }, id: 'writer-b:1', text: 'beta' }],
        members: [{ name: 'beta' }],
        sessions: {},
        syncRevision: 0,
        watermarks: {}
      }
    } as unknown as Record<string, GroupChat>)

    room.chat.scheduleGroupChatServerSync(room.chat.$groupChats.get(), { changedRooms: ['Shared'] })
    await drain(() => (room.gateway.uiMetaRevisions['hermes-bots-groups'] || 0) < 2, 60)

    const stored = room.gateway.uiMeta['hermes-bots-groups'] as {
      rooms: Record<string, { log: GroupMessage[] }>
    }

    expect(room.gateway.uiMetaRevisions['hermes-bots-groups']).toBe(2)
    expect(stored.rooms['name:Shared'].log.map(entry => entry.id).sort()).toEqual(['writer-a:1', 'writer-b:1'])
    expect(room.gateway.rpcFor('profiles.configure')).toHaveLength(2)
  })

  it('fans a room write out to every reachable default-profile gateway', async () => {
    const room = await loadRoom()
    const remote: { connectionId: string; method: string }[] = []
    host.profileRoutes = async () => [
      { connectionId: 'gw-a', profile: 'default' },
      { connectionId: 'gw-b', profile: 'default' },
      { connectionId: 'gw-b', profile: 'other' }
    ]

    host.requestProfile = async (route: { connectionId: string }, method: string) => {
      remote.push({ connectionId: route.connectionId, method })

      if (method === 'profiles.list') {
        return { profiles: [{ name: 'default', ui_meta: {}, ui_meta_revisions: {} }] }
      }

      if (method === 'profiles.configure') {
        return { applied: { ui_meta: true, ui_meta_revisions: { 'hermes-bots-groups': 1 } } }
      }

      return {}
    }

    room.chat.$groupChats.set({
      Shared: {
        log: [{ at: 1, from: { kind: 'user', name: 'You' }, id: 'w1', text: 'hi' }],
        members: [{ name: 'research' }],
        sessions: {},
        syncRevision: 0,
        watermarks: {}
      }
    } as unknown as Record<string, GroupChat>)

    room.chat.scheduleGroupChatServerSync(room.chat.$groupChats.get(), { changedRooms: ['Shared'] })
    await drain(() => remote.filter(entry => entry.method === 'profiles.configure').length < 2, 80)

    const configured = new Set(
      remote.filter(entry => entry.method === 'profiles.configure').map(entry => entry.connectionId)
    )

    expect(configured.has('gw-a')).toBe(true)
    expect(configured.has('gw-b')).toBe(true)
  })
})

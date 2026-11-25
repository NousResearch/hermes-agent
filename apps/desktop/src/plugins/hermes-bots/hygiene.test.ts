import { describe, expect, it, vi } from 'vitest'

import type * as data from './data'
import type * as groupChat from './group-chat'
import { createGroupGateway, scriptedStorage } from './group-test-utils'
import type * as hygiene from './hygiene'
import type { GroupChat, GroupMember, RosterRow } from './types'

// #93492 root fix: deleting a connection sweeps/annotates the persisted
// group-chat member rows that referenced it (registry 'removed' push), and a
// hydrate-time pass annotates rows orphaned before the sweep existed. Rows are
// marked (sourceMissing → the existing 'Gateway removed' degraded state),
// never silently deleted — the poisoned row lives in plugin storage, so a hard
// delete would take the user's membership with it.

const { host } = vi.hoisted(() => ({ host: {} as Record<string, unknown> }))

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')

  return pluginSdkMock(host)
})

interface Rooms {
  chat: typeof groupChat
  data: typeof data
  hygiene: typeof hygiene
  writes: Map<string, unknown>
}

async function load(): Promise<Rooms> {
  vi.resetModules()
  const gateway = createGroupGateway()

  for (const key of Object.keys(host)) {
    delete host[key]
  }

  Object.assign(host, gateway.host)

  const [chat, data, hygiene, shared] = await Promise.all([
    import('./group-chat'),
    import('./data'),
    import('./hygiene'),
    import('./shared')
  ])

  shared.setPluginCtx(scriptedStorage(gateway.storage))

  return { chat, data, hygiene, writes: gateway.storage }
}

function member(overrides: Partial<GroupMember> = {}): GroupMember {
  return {
    connectionId: 'conn-gone',
    handle: 'dixie',
    name: 'dixie',
    remoteSource: true,
    route: { connectionId: 'conn-gone', mode: 'remote', profile: 'dixie', targetProfile: 'dixie' },
    sourceScoped: true,
    ...overrides
  }
}

describe('removed-connection sweep', () => {
  it('annotates matching members and persists; other rows are untouched', async () => {
    const { chat, data, writes } = await load()
    chat.$groupChats.set({
      Crew: {
        log: [{ at: 1, from: { kind: 'user', name: 'You' }, id: '1', text: 'hi' }],
        members: [
          member(),
          member({
            connectionId: 'conn-live',
            name: 'bob',
            route: { connectionId: 'conn-live', mode: 'remote', profile: 'bob', targetProfile: 'bob' }
          })
        ],
        watermarks: {}
      }
    })

    expect(chat.sweepGroupChatMembersForRemovedConnection('conn-gone')).toBe(true)

    const room = chat.$groupChats.get().Crew
    const swept = room.members?.find(row => row.name === 'dixie')
    const kept = room.members?.find(row => row.name === 'bob')

    // Annotated, not deleted: identity survives, marked degraded.
    expect(room.members).toHaveLength(2)
    expect(swept?.sourceMissing).toBe(true)
    expect(swept?.sourceReachable).toBe(false)
    expect(kept?.sourceMissing).toBeUndefined()
    // The degraded mark renders as the existing 'Gateway removed' state.
    expect(data.botSourceStatus(swept).label).toBe('Gateway removed')
    // Persisted so the fix survives restarts (the poisoned row lived in storage).
    expect(writes.has('group-chats')).toBe(true)
  })

  it('is idempotent and ignores blank ids', async () => {
    const { chat } = await load()
    chat.$groupChats.set({ Crew: { log: [], members: [member()], watermarks: {} } })

    expect(chat.sweepGroupChatMembersForRemovedConnection('')).toBe(false)
    expect(chat.sweepGroupChatMembersForRemovedConnection('conn-gone')).toBe(true)
    // Already annotated: nothing left to change.
    expect(chat.sweepGroupChatMembersForRemovedConnection('conn-gone')).toBe(false)
  })
})

describe('hydrate-time annotate', () => {
  it('marks lost-connectionId rows even without a registry', async () => {
    const { data, hygiene } = await load()

    const rooms: Record<string, GroupChat> = {
      Crew: {
        log: [],
        // The exact persisted shape from #93492: remoteSource kept, connectionId
        // NULLED. The descriptor type models it as absent, but what older builds
        // actually wrote to disk is an explicit null — that is the shape the
        // annotate pass has to recognise.
        members: [
          { connectionId: null, handle: 'halakukhan', name: 'halakukhan', remoteSource: true } as unknown as GroupMember
        ],
        watermarks: {}
      }
    }

    const { rooms: next, changed } = hygiene.annotateOrphanedGroupChatMembers(rooms, null)

    expect(changed).toBe(true)
    expect(next.Crew.members?.[0].sourceMissing).toBe(true)
    expect(data.botSourceStatus(next.Crew.members?.[0]).label).toBe('Gateway removed')
  })

  it('with a live registry, marks members on dead connections and keeps live and local ones', async () => {
    const { hygiene } = await load()

    const rooms: Record<string, GroupChat> = {
      Crew: {
        log: [],
        members: [
          member(),
          member({
            connectionId: 'conn-live',
            name: 'bob',
            route: { connectionId: 'conn-live', mode: 'remote', profile: 'bob', targetProfile: 'bob' }
          }),
          { name: 'local-pal' }
        ],
        watermarks: {}
      }
    }

    const { rooms: next, changed } = hygiene.annotateOrphanedGroupChatMembers(rooms, new Set(['conn-live']))
    const seated = (name: string) => next.Crew.members?.find(row => row.name === name)

    expect(changed).toBe(true)
    expect(seated('dixie')?.sourceMissing).toBe(true)
    expect(seated('bob')?.sourceMissing).toBeUndefined()
    expect(seated('local-pal')?.sourceMissing).toBeUndefined()
  })

  it('without a registry, touches only the unresolvable-route shape', async () => {
    const { hygiene } = await load()

    // conn-gone still has an id; without a registry we cannot prove it dead.
    const { changed } = hygiene.annotateOrphanedGroupChatMembers(
      { Crew: { log: [], members: [member()], watermarks: {} } },
      null
    )

    expect(changed).toBe(false)
  })
})

describe('render-path callers', () => {
  it('degrade on an orphaned row instead of throwing', async () => {
    const { chat } = await load()
    const [membership, routing] = await Promise.all([import('./group-membership'), import('./routing')])

    // Persisted-on-disk shape, nulled connectionId included (see above).
    const orphaned = {
      connectionId: null,
      handle: 'halakukhan',
      name: 'halakukhan',
      remoteSource: true
    } as unknown as RosterRow

    // botWorkspaceOwnerKey (sidebar sync / Bots home / context menus)
    expect(routing.botWorkspaceOwnerKey(orphaned)).toBe('bot:halakukhan')
    // setBotsWorkspaceOwner (sidebar visibility listener)
    expect(() => routing.setBotsWorkspaceOwner('bot:halakukhan', orphaned)).not.toThrow()
    // durableGroupChatMembers (every group send / roster refresh)
    expect(() => membership.durableGroupChatMembers([orphaned])).not.toThrow()

    const durable = membership.durableGroupChatMembers([
      { ...orphaned, sourceMissing: true, sourceReachable: false }
    ])[0]

    // The degraded mark survives the durable rebuild.
    expect(durable.sourceMissing).toBe(true)
    expect(chat.$groupChats.get()).toEqual({})
  })
})

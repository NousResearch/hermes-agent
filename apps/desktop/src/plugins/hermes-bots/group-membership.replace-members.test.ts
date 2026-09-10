/**
 * Replacing a room's whole member set writes to TWO places: the durable room
 * roster (authoritative) and each local bot's ui_meta `groups` list (the
 * compatibility and discovery projection). They disagree if a write fails
 * halfway — and a disagreement is exactly the bug editable membership exists
 * to fix, a removed bot that stays seated.
 *
 * So the contract here is transactional, not incremental: either every
 * profile's metadata lands and the room roster is swapped, or nothing is left
 * changed and the caller is told which profiles failed.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

import type * as data from './data'
import type * as groupChat from './group-chat'
import type * as groupMembership from './group-membership'
import { createGroupGateway, scriptedStorage } from './group-test-utils'
import type { GroupChat, RosterRow } from './types'

const { host } = vi.hoisted(() => ({ host: {} as Record<string, unknown> }))

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')

  return pluginSdkMock(host)
})

interface Modules {
  chat: typeof groupChat
  data: typeof data
  membership: typeof groupMembership
}

/** `failConfigure` names the profiles whose `profiles.configure` answers with
 *  an explicit "did not apply" — the only shape saveBotMeta reads as `failed`
 *  (a rejected call is the old-gateway `unsupported` path instead). */
async function load(failConfigure: Record<string, true> = {}): Promise<Modules> {
  vi.resetModules()
  const gateway = createGroupGateway({ failConfigureFor: failConfigure })

  for (const key of Object.keys(host)) {
    delete host[key]
  }

  Object.assign(host, gateway.host)

  const [chat, dataModule, membership, shared] = await Promise.all([
    import('./group-chat'),
    import('./data'),
    import('./group-membership'),
    import('./shared')
  ])

  shared.setPluginCtx(scriptedStorage(gateway.storage))

  return { chat, data: dataModule, membership }
}

let modules: Modules

function rooms(map: Record<string, Partial<GroupChat>>): Record<string, GroupChat> {
  return Object.fromEntries(Object.entries(map).map(([name, room]) => [name, { log: [], watermarks: {}, ...room }]))
}

function bot(name: string, extra: Partial<RosterRow> = {}): RosterRow {
  return { name, ...extra } as RosterRow
}

beforeEach(async () => {
  modules = await load()
})

describe('seating a roomId-backed roster', () => {
  it('treats the stored members as authoritative, so a removed bot does not come back', async () => {
    // The removed bot's own ui_meta still names the group until its write
    // lands. The legacy union would re-seat it from that stale meta.
    modules.chat.$groupChats.set(rooms({ Ops: { members: [{ name: 'pm' }], roomId: 'room-1' } }))

    const seated = modules.membership.groupChatMemberBots('Ops', [bot('pm'), bot('dropped')], {
      dropped: { groups: ['Ops'] },
      pm: { groups: ['Ops'] }
    })

    expect(seated.map(member => member.name)).toEqual(['pm'])
  })

  it('still unions bot-meta for a legacy room that carries no roomId', () => {
    modules.chat.$groupChats.set(rooms({ Ops: { members: [{ name: 'pm' }] } }))

    const seated = modules.membership.groupChatMemberBots('Ops', [bot('pm'), bot('alsoIn')], {
      alsoIn: { groups: ['Ops'] },
      pm: { groups: ['Ops'] }
    })

    expect(seated.map(member => member.name).sort()).toEqual(['alsoIn', 'pm'])
  })

  it('skips a nameless stored descriptor instead of seating it as a ghost', () => {
    // botRosterKey synthesizes `legacy::default` for a nameless row, which
    // would seat one malformed descriptor and swallow every later one.
    modules.chat.$groupChats.set(
      rooms({ Ops: { members: [{ name: '' }, { name: '' }, { name: 'pm' }] as never, roomId: 'room-1' } })
    )

    const seated = modules.membership.groupChatMemberBots('Ops', [bot('pm')], {})

    expect(seated.map(member => member.name)).toEqual(['pm'])
  })
})

describe('replacing a member set that succeeds', () => {
  it('writes the room roster and adds the group to each seated profile', async () => {
    modules.data.$botMeta.set({ pm: {}, scout: {} })

    await modules.membership.replaceGroupChatMembers('Ops', [bot('pm'), bot('scout')])

    expect(modules.chat.$groupChats.get().Ops.members?.map(member => member.name)).toEqual(['pm', 'scout'])
    expect(modules.membership.botGroups(modules.data.$botMeta.get().pm)).toEqual(['Ops'])
    expect(modules.membership.botGroups(modules.data.$botMeta.get().scout)).toEqual(['Ops'])
  })

  it('un-seats a profile that still claims the group but is no longer selected', async () => {
    modules.data.$botMeta.set({ dropped: { groups: ['Ops'] }, pm: { groups: ['Ops'] } })

    await modules.membership.replaceGroupChatMembers('Ops', [bot('pm')])

    expect(modules.membership.botGroups(modules.data.$botMeta.get().dropped)).toEqual([])
    expect(modules.chat.$groupChats.get().Ops.members?.map(member => member.name)).toEqual(['pm'])
  })

  it('de-duplicates by seat key and skips a nameless row', async () => {
    const seated = await modules.membership.replaceGroupChatMembers('Ops', [bot('pm'), bot('pm'), bot('')])

    expect(seated.map(member => member.name)).toEqual(['pm'])
  })

  it('rejects an empty set and one past the cap before writing anything', async () => {
    modules.data.$botMeta.set({ pm: {} })

    await expect(modules.membership.replaceGroupChatMembers('Ops', [])).rejects.toThrow(/require/)
    await expect(
      modules.membership.replaceGroupChatMembers(
        'Ops',
        Array.from({ length: 40 }, (_, index) => bot(`bot${index}`))
      )
    ).rejects.toThrow(/require/)
    expect(modules.chat.$groupChats.get().Ops).toBeUndefined()
    expect(modules.membership.botGroups(modules.data.$botMeta.get().pm)).toEqual([])
  })

  it('does not roll back an old gateway that cannot persist remotely', async () => {
    // A rejected configure is `unsupported`, not `failed`: the local write is
    // still saved and the room must be updated, exactly as every other
    // metadata edit in the plugin behaves on an old gateway.
    const legacy = await load()
    delete (host as { request?: unknown }).request

    modules = legacy
    modules.data.$botMeta.set({ pm: {} })

    await modules.membership.replaceGroupChatMembers('Ops', [bot('pm')])

    expect(modules.chat.$groupChats.get().Ops.members?.map(member => member.name)).toEqual(['pm'])
  })
})

describe('replacing a member set where a write fails', () => {
  it('leaves the room roster untouched and names the failing profile', async () => {
    modules = await load({ scout: true })
    modules.data.$botMeta.set({ pm: {}, scout: {} })

    // The label is the bot's DISPLAY name, not its slug — the message is for
    // a person reading a toast.
    await expect(modules.membership.replaceGroupChatMembers('Ops', [bot('pm'), bot('scout')])).rejects.toThrow(
      /Scout: server rejected the write/
    )
    expect(modules.chat.$groupChats.get().Ops).toBeUndefined()
  })

  it('restores the exact metadata snapshot, since saveBotMeta only merges', async () => {
    modules = await load({ scout: true })
    // `pm` had NO group fields at all. A merge-based rollback would leave it
    // carrying a synthetic empty projection instead of its original shape.
    const before = { pm: {}, scout: {} }
    modules.data.$botMeta.set(before)

    await expect(modules.membership.replaceGroupChatMembers('Ops', [bot('pm'), bot('scout')])).rejects.toThrow()
    expect(modules.data.$botMeta.get()).toEqual(before)
  })

  it('reports an unconfirmed rollback distinctly from a clean one', async () => {
    // Both profiles refuse: `pm` refuses its write AND its rollback, so the
    // caller must be told remote metadata may be inconsistent rather than
    // being reassured that everything was undone.
    modules = await load({ pm: true, scout: true })
    modules.data.$botMeta.set({ pm: {}, scout: {} })

    // Neither write was confirmed persisted, so there is nothing to roll back
    // and the message stays the clean one.
    await expect(modules.membership.replaceGroupChatMembers('Ops', [bot('pm'), bot('scout')])).rejects.toThrow(
      /changes were rolled back/
    )
  })
})

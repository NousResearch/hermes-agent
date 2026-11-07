import { beforeEach, describe, expect, it, vi } from 'vitest'

import type * as data from './data'
import type * as groupChat from './group-chat'
import type * as members from './group-chat-view-members'
import type * as groupMembership from './group-membership'
import { createGroupGateway, drain, runTimersInline, scriptedStorage } from './group-test-utils'
import type { RosterRow } from './types'

// A membership edit is only as true as what each bot's OWN machine holds:
// a write that dropped mid-flight is not an older gateway, an undo restores
// only what its edit changed, and an undo that did not land is never
// reported as "nothing was changed".

const { host } = vi.hoisted(() => ({ host: {} as Record<string, unknown> }))

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')

  return pluginSdkMock(host)
})

interface Room {
  chat: typeof groupChat
  data: typeof data
  members: typeof members
  membership: typeof groupMembership
}

async function loadRoom(): Promise<Room> {
  vi.resetModules()
  const gateway = createGroupGateway()

  for (const key of Object.keys(host)) {
    delete host[key]
  }

  Object.assign(host, gateway.host)

  const [chat, d, m, membership, shared] = await Promise.all([
    import('./group-chat'),
    import('./data'),
    import('./group-chat-view-members'),
    import('./group-membership'),
    import('./shared')
  ])

  shared.setPluginCtx(scriptedStorage(gateway.storage))

  return { chat, data: d, members: m, membership }
}

/** One remote machine's profile.yaml for its `default` profile, with the
 *  gateway's ui_meta CAS. `fault` runs before each call and may throw. */
class RemoteMachine {
  groups: string[]
  revision = 1
  fault: (method: string) => Promise<void> | void = () => undefined

  constructor(groups: string[]) {
    this.groups = groups
  }

  async handle(method: string, params: Record<string, any>) {
    await this.fault(method)

    if (method === 'profiles.list') {
      return {
        profiles: [
          {
            name: 'default',
            ui_meta: { 'hermes-bots': { groups: this.groups } },
            ui_meta_revisions: { 'hermes-bots': this.revision }
          }
        ]
      }
    }

    if (method === 'profiles.configure') {
      if (params.ui_meta_expected_revisions?.['hermes-bots'] !== this.revision) {
        return { applied: { ui_meta: false, ui_meta_conflicts: { 'hermes-bots': { actual: this.revision } } } }
      }

      this.groups = [...(params.ui_meta['hermes-bots'].groups || [])]
      this.revision += 1

      return { applied: { ui_meta: true } }
    }

    return {}
  }
}

const transportClosed = () => new Error('backend transport closed')

function remoteBot(connectionId: string, title: string): RosterRow {
  return {
    connectionId,
    connectionKind: 'remote',
    connectionLabel: connectionId,
    display_name: title,
    name: 'default',
    remoteSource: true,
    route: { connectionId, mode: 'remote', profile: 'default', targetProfile: 'default' },
    sourceScoped: true
  } as RosterRow
}

const programmer: RosterRow = { name: 'programmer' }
const reviewer: RosterRow = { name: 'reviewer' }

function serve(machines: Record<string, RemoteMachine>) {
  host.requestProfile = (route: { connectionId: string }, method: string, params: Record<string, any>) =>
    machines[route.connectionId].handle(method, params)
}

function seed(room: Room, remotes: RosterRow[], remoteGroups: string[], seated: RosterRow[]) {
  room.data.$lastRoster.set([programmer, reviewer, ...remotes])
  room.data.$botMeta.set({
    programmer: { groups: ['Core'] },
    reviewer: { groups: ['Core'] },
    ...Object.fromEntries(remotes.map(bot => [room.data.botMetaKey(bot), { groups: remoteGroups }]))
  })
  room.chat.updateGroupChat(
    'Core',
    r => ({ ...r, members: room.membership.durableGroupChatMembers(seated), syncRevision: 3 }),
    { sync: false }
  )
}

const seatedKeys = (room: Room) =>
  room.membership
    .groupChatMemberBots('Core', room.data.$lastRoster.get(), room.data.$botMeta.get())
    .map(bot => `${bot.connectionId || 'local'}::${bot.name}`)
    .sort()

beforeEach(() => {
  runTimersInline()
})

describe('a membership write its machine dropped after the read', () => {
  it('fails the removal and keeps the seat: the machine still names the room', async () => {
    const room = await loadRoom()
    const homelab = remoteBot('homelab', 'Homelab Hermes')
    const machine = new RemoteMachine(['Core'])
    serve({ homelab: machine })
    seed(room, [homelab], ['Core'], [programmer, reviewer, homelab])

    // The read answers; the backend goes away before the write lands, and is back for the undo.
    let dropped = false

    machine.fault = method => {
      if (method === 'profiles.configure' && !dropped) {
        dropped = true
        throw transportClosed()
      }
    }

    await expect(room.members.setGroupMembership(homelab, 'Core', false)).rejects.toThrow(/Homelab Hermes/)

    expect(machine.groups).toEqual(['Core'])
    expect(seatedKeys(room)).toEqual(['homelab::default', 'local::programmer', 'local::reviewer'])
  })

  it('still treats a gateway without the method as the older-gateway fallback', async () => {
    const room = await loadRoom()
    const homelab = remoteBot('homelab', 'Homelab Hermes')
    const machine = new RemoteMachine(['Core'])
    serve({ homelab: machine })
    seed(room, [homelab], ['Core'], [programmer, reviewer, homelab])

    machine.fault = method => {
      if (method === 'profiles.configure') {
        throw Object.assign(new Error('unknown method: profiles.configure'), { code: -32601 })
      }
    }

    await expect(room.data.saveBotMeta(homelab, { groups: [] })).resolves.toMatchObject({
      serverOutcome: 'unsupported'
    })
  })
})

describe('overlapping edits of one bot', () => {
  it('an older failed join never erases a newer acknowledged one', async () => {
    const room = await loadRoom()
    const homelab = remoteBot('homelab', 'Homelab Hermes')
    const machine = new RemoteMachine([])
    serve({ homelab: machine })
    seed(room, [homelab], [], [programmer, reviewer])

    // Join A's read is held, then fails on a closed transport; everything after it answers.
    const held: { fail?: (error: Error) => void } = {}
    let reads = 0

    machine.fault = method => {
      if (method === 'profiles.list' && ++reads === 1) {
        return new Promise<void>((_resolve, reject) => {
          held.fail = reject
        })
      }
    }

    const joinA = room.members.setGroupMembership(homelab, 'A', true)
    const joinB = room.members.setGroupMembership(homelab, 'B', true)
    await drain(() => !held.fail)
    // The newer edit waits for the older one instead of reading around it.
    expect(reads).toBe(1)
    held.fail?.(transportClosed())

    await expect(joinA).rejects.toThrow(/Homelab Hermes/)
    await expect(joinB).resolves.toBeUndefined()

    expect(machine.groups).toEqual(['B'])
    expect(room.data.$botMeta.get()[room.data.botMetaKey(homelab)].groups).toEqual(['B'])
  })
})

describe('an undo that does not land', () => {
  it('is reported as such, and the room shows the seat its machine kept', async () => {
    const room = await loadRoom()
    const remoteA = remoteBot('remote-a', 'Atlas')
    const remoteB = remoteBot('remote-b', 'Borealis')
    const machineA = new RemoteMachine([])
    const machineB = new RemoteMachine([])
    serve({ 'remote-a': machineA, 'remote-b': machineB })
    seed(room, [remoteA, remoteB], [], [programmer, reviewer])

    // A takes the join; then both machines drop: B never answers, and A cannot take the undo.
    let aDown = false

    machineA.fault = () => {
      if (aDown) {
        throw transportClosed()
      }
    }

    machineB.fault = () => {
      aDown = true
      throw transportClosed()
    }

    const save = room.members.setGroupChatMembers('Core', [programmer, reviewer, remoteA, remoteB])

    const error = await save.then(
      () => null,
      (reason: Error) => reason.message
    )

    expect(error).toMatch(/Atlas/)
    expect(error).toMatch(/remote-a/)
    expect(error).not.toMatch(/Nothing was changed/)
    expect(machineA.groups).toEqual(['Core'])
    expect(seatedKeys(room)).toEqual(['local::programmer', 'local::reviewer', 'remote-a::default'])
    expect(
      room.chat.$groupChats.get().Core.members?.map(member => `${member.connectionId || 'local'}::${member.name}`)
    ).toContain('remote-a::default')
  })
})

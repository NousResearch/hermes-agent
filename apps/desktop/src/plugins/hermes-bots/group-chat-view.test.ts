import { beforeEach, describe, expect, it, vi } from 'vitest'

import type * as data from './data'
import type * as groupChat from './group-chat'
import type * as groupChatView from './group-chat-view'
import type * as groupMembership from './group-membership'
import type * as groupPanes from './group-panes'
import { createGroupGateway, drain, runTimersInline, scriptedStorage } from './group-test-utils'
import type { ScriptedGateway } from './group-test-utils'
import type { GroupChat, RosterRow } from './types'

// The room surface's two lifecycle mutations — opening a room into the MAIN
// window, and disbanding one — plus the ordering rules that keep the in-pane
// fallback from painting a duplicate beside the main tab.

const { host } = vi.hoisted(() => ({ host: {} as Record<string, unknown> }))

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')

  return pluginSdkMock(host)
})

interface Room {
  chat: typeof groupChat
  data: typeof data
  gateway: ScriptedGateway
  membership: typeof groupMembership
  panes: typeof groupPanes
  view: typeof groupChatView
}

async function loadRoom(): Promise<Room> {
  vi.resetModules()
  const gateway = createGroupGateway()

  for (const key of Object.keys(host)) {
    delete host[key]
  }

  Object.assign(host, gateway.host)

  const [chat, data, membership, panes, view, shared] = await Promise.all([
    import('./group-chat'),
    import('./data'),
    import('./group-membership'),
    import('./group-panes'),
    import('./group-chat-view'),
    import('./shared')
  ])

  shared.setPluginCtx(scriptedStorage(gateway.storage))

  return { chat, data, gateway, membership, panes, view }
}

const durable = (room: Room) => (room.gateway.storage.get('group-chats') || {}) as Record<string, GroupChat>

beforeEach(() => {
  runTimersInline()
})

describe('opening a room', () => {
  it('gives member replies a bordered assistant flashcard distinct from user prompts', async () => {
    const room = await loadRoom()
    const agent = room.view.groupMessageCardPresentation(false)
    const user = room.view.groupMessageCardPresentation(true)

    expect(agent.slot).toBe('bot-group-agent-card')
    expect(agent.className).toContain('border-l-4')
    expect(agent.className).toContain('ui-chat-bubble-opaque-background')
    expect(user.slot).toBe('bot-group-user-card')
    expect(user.className).not.toContain('border-l-4')
  })

  it('follows the main-window tab open and close', async () => {
    const room = await loadRoom()
    let onClose: () => void = () => undefined

    host.openWorkspace = (_id: string, options: { onClose: () => void }) => {
      onClose = options.onClose

      return () => onClose()
    }

    room.view.openGroupChat('Core')

    expect(room.chat.$groupChatWorkspace.get()).toBe('Core')
    // #89788: the main tab owns the room, so the pane keeps its roster.
    expect(room.panes.shouldRenderGroupChatInPane('Core')).toBe(false)

    onClose()

    expect(room.chat.$groupChatWorkspace.get()).toBeNull()
    expect(room.panes.shouldRenderGroupChatInPane('Core')).toBe(true)
  })

  it('keeps the in-pane fallback on older hosts and when the door throws', async () => {
    const older = await loadRoom()

    older.view.openGroupChat('Core')

    expect(older.chat.$groupChatWorkspace.get()).toBe('Core')
    expect(older.panes.shouldRenderGroupChatInPane('Core')).toBe(true)

    const failed = await loadRoom()

    host.openWorkspace = () => {
      throw new Error('workspace unavailable')
    }

    failed.view.openGroupChat('Ops')

    expect(failed.chat.$groupChatWorkspace.get()).toBe('Ops')
    expect(failed.panes.shouldRenderGroupChatInPane('Ops')).toBe(true)
  })

  it('records main-tab ownership before the selection atom paints (#89788 follow-up)', async () => {
    const room = await loadRoom()
    // Simulate a BotsPane render racing the open: sample the gate at the
    // moment the selection atom flips. If the tab were recorded after the atom
    // set, this probe would observe selected-but-unowned and the in-pane
    // duplicate would paint beside the main tab.
    let gateAtSelection: boolean | null = null

    const unsubscribe = room.chat.$groupChatWorkspace.listen(value => {
      if (value === 'Core' && gateAtSelection === null) {
        gateAtSelection = room.panes.shouldRenderGroupChatInPane('Core')
      }
    })

    host.openWorkspace = () => () => undefined

    room.view.openGroupChat('Core')
    unsubscribe()

    expect(gateAtSelection).toBe(false)
  })

  it('does not let an older group closing clear the newer selection', async () => {
    const room = await loadRoom()
    host.openWorkspace = () => () => undefined

    room.view.openGroupChat('Core')
    room.view.openGroupChat('Ops')
    room.panes.closeGroupChatMainTab('Core')

    expect(room.chat.$groupChatWorkspace.get()).toBe('Ops')
  })
})

describe('disband', () => {
  it('removes only this membership, room log, workspace and needs-you state', async () => {
    const room = await loadRoom()
    room.chat.$groupChats.set({
      Gone: { log: [{ at: 2, from: { kind: 'user', name: 'You' }, id: 'g1', text: 'hello goners' }], watermarks: {} },
      Keep: {
        log: [{ at: 1, from: { kind: 'user', name: 'You' }, id: 'k1', text: 'hello keepers' }],
        members: [{ connectionId: 'remote-1', name: 'remote', remoteSource: true, sourceScoped: true }],
        watermarks: {}
      }
    } as unknown as Record<string, GroupChat>)
    room.data.$botMeta.set({
      builder: { group: 'Gone', groups: ['Gone', 'Keep'] },
      research: { group: 'Keep', groups: ['Keep'] }
    })
    room.chat.$groupChatWorkspace.set('Gone')
    room.chat.$groupNeedsYou.set({ Gone: true, Keep: true })

    await room.view.disbandGroupChat('Gone', [{ name: 'builder' }])

    // Room state: gone from the atom (no running drive, so no tombstone).
    expect(room.chat.$groupChats.get().Gone).toBeUndefined()
    expect(room.chat.$groupChats.get().Keep).toBeTruthy()
    // The open room view closed; needs-you cleared for the disbanded room only.
    expect(room.chat.$groupChatWorkspace.get()).toBeNull()
    expect(room.chat.$groupNeedsYou.get().Gone).toBeUndefined()
    expect(room.chat.$groupNeedsYou.get().Keep).toBe(true)
    // Disband removes only this membership; other groups survive.
    expect(room.data.$botMeta.get().builder.groups).toEqual(['Keep'])
    expect(room.data.$botMeta.get().builder.group).toBe('Keep')
    expect(room.data.$botMeta.get().research.groups).toEqual(['Keep'])
    expect('Gone' in durable(room)).toBe(false)
    expect(durable(room).Keep.members).toHaveLength(1)
    expect(durable(room).Keep.members?.[0].connectionId).toBe('remote-1')
  })

  it('cannot leave a metadata-only group row behind when the rendered roster is empty', async () => {
    const room = await loadRoom()
    room.data.$botMeta.set({ builder: { group: 'Remote', groups: ['Remote'] } })
    room.chat.$groupChats.set({
      Remote: { log: [], members: [], running: false, sessions: {}, watermarks: {} }
    } as unknown as Record<string, GroupChat>)

    await room.view.disbandGroupChat('Remote', [])

    expect(room.chat.$groupChats.get().Remote).toBeUndefined()
    expect(room.data.$botMeta.get().builder.groups).toEqual([])
    expect(room.data.$botMeta.get().builder.group).toBeNull()
    // Stale bot metadata cannot reconstruct a deleted zero-member row.
    expect(room.membership.groupChatNames(room.data.$botMeta.get(), room.chat.$groupChats.get())).toEqual([])
  })

  it('recovers the exact source-qualified metadata owner from an empty roster', async () => {
    const room = await loadRoom()

    const remote: RosterRow = {
      connectionId: 'remote-1',
      connectionKind: 'remote',
      name: 'builder',
      remoteSource: true,
      route: { connectionId: 'remote-1', mode: 'remote', profile: 'builder', targetProfile: 'builder' },
      sourceScoped: true
    }

    room.data.$lastRoster.set([remote])
    room.data.$botMeta.set({
      builder: { group: 'Keep', groups: ['Keep'] },
      'remote-1::builder': { group: 'Remote', groups: ['Remote'] }
    })
    room.chat.$groupChats.set({
      Remote: { log: [], members: [], running: false, sessions: {}, watermarks: {} }
    } as unknown as Record<string, GroupChat>)

    await room.view.disbandGroupChat('Remote', [])

    expect(room.data.$botMeta.get()['remote-1::builder'].groups).toEqual([])
    expect(room.data.$botMeta.get()['remote-1::builder'].group).toBeNull()
    // Same-named local metadata is untouched.
    expect(room.data.$botMeta.get().builder.groups).toEqual(['Keep'])
    expect(room.data.$botMeta.get().builder.group).toBe('Keep')
  })

  it('skips source-qualified remote members instead of mutating same-named local metadata', async () => {
    const room = await loadRoom()
    room.data.$botMeta.set({ builder: { group: 'Keep', groups: ['Keep'] } })

    await room.view.disbandGroupChat('Remote', [
      { connectionId: 'remote-1', name: 'builder', remoteSource: true, sourceScoped: true }
    ])

    expect(room.data.$botMeta.get().builder.groups).toEqual(['Keep'])
    expect(room.data.$botMeta.get().builder.group).toBe('Keep')
    expect(room.data.$botMeta.get()['[object Object]']).toBeUndefined()
  })

  it('leaves an epoch-bumped empty tombstone while a drive is mid-turn', async () => {
    const room = await loadRoom()
    room.chat.$groupChats.set({
      Live: {
        epoch: 3,
        log: [{ at: 1, from: { kind: 'user', name: 'You' }, id: 'l1', text: 'kick off' }],
        running: true,
        watermarks: {}
      }
    } as unknown as Record<string, GroupChat>)

    await room.view.disbandGroupChat('Live', [{ name: 'research' }])

    const tomb = room.chat.$groupChats.get().Live

    expect(tomb).toBeTruthy()
    expect(tomb.log).toHaveLength(0)
    expect(tomb.running).toBe(false)
    // Epoch bumped so the loop bails at its member boundary; flagged so
    // persistence and name-dedup skip it.
    expect(tomb.epoch).toBe(4)
    expect(tomb.tombstone).toBe(true)
    expect('Live' in durable(room)).toBe(false)

    // Regression (#90028 live E2E): updateGroupChat persists the WHOLE atom
    // map — an unrelated room write while the tombstone lingers must not
    // smuggle it into durable storage, and the disbanded name must be
    // immediately reusable (uniqueGroupChatName would suffix it otherwise).
    room.chat.updateGroupChat('Other', current => {
      current.log.push({ at: Date.now(), from: { kind: 'user', name: 'You' }, id: 'o1', text: 'x', thread: 't' })

      return current
    })

    expect('Other' in durable(room)).toBe(true)
    expect('Live' in durable(room)).toBe(false)
    expect(room.chat.uniqueGroupChatName('Live', new Set(room.membership.liveGroupChatNames()))).toBe('Live')
  })

  it('drops the disbanded room from the gateway mirror', async () => {
    const room = await loadRoom()
    room.chat.$groupChats.set({
      Gone: { log: [{ at: 1, from: { kind: 'user', name: 'You' }, id: 'g1', text: 'bye' }], watermarks: {} }
    } as unknown as Record<string, GroupChat>)

    await room.view.disbandGroupChat('Gone', [])
    await drain(() => room.gateway.rpcFor('profiles.configure').length < 1, 50)

    const configure = room.gateway.rpcFor('profiles.configure').at(-1)

    const envelope = (configure?.params.ui_meta as Record<string, { deleted?: Record<string, number> }>)[
      'hermes-bots-groups'
    ]

    expect(envelope.deleted?.['name:Gone']).toBeGreaterThan(0)
  })
})

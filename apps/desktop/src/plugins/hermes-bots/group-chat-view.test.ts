import type * as HermesSdk from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { createElement, Fragment } from 'react'
import type { ComponentProps, ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { CANONICAL_GROUP_LOCALES } from './canonical-group-locales'
import type * as data from './data'
import type * as groupChat from './group-chat'
import type * as groupChatView from './group-chat-view'
import type * as groupMembership from './group-membership'
import type * as groupPanes from './group-panes'
import { createGroupGateway, drain, runTimersInline, scriptedStorage } from './group-test-utils'
import type { ScriptedGateway } from './group-test-utils'
import { translateBots } from './i18n-test-helper'
import type { GroupChat, RosterRow } from './types'

// The room surface's two lifecycle mutations — opening a room into the MAIN
// window, and disbanding one — plus the ordering rules that keep the in-pane
// fallback from painting a duplicate beside the main tab.

const { host } = vi.hoisted(() => ({ host: {} as Record<string, unknown> }))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const original = await importOriginal<typeof HermesSdk>()
  const { pluginSdkMock } = await import('./group-test-utils')
  const children = ({ children }: { children?: ReactNode }) => createElement(Fragment, null, children)
  const button = (props: ComponentProps<'button'>) => createElement('button', { type: 'button', ...props })

  return {
    ...original, ...await pluginSdkMock(host),
    Badge: children, Dialog: children, DialogContent: children, DialogDescription: children,
    DialogFooter: children, DialogHeader: children, DialogTitle: children, Tip: children,
    Button: button, RowButton: button, Input: (props: ComponentProps<'input'>) => createElement('input', props),
    SearchField: () => null, Codicon: () => null,
    useI18n: () => ({ t: { common: { cancel: 'Cancel' } } }), usePluginI18n: () => translateBots
  }
})
vi.mock('./group-chat-parts', () => ({ GroupImageControls: () => null }))

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
afterEach(() => cleanup())

describe('opening a room', () => {
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

describe('gateway room tabs', () => {
  it('names a gateway room tab after the room, and keeps its binding through a rename', async () => {
    const room = await loadRoom()
    const { $canonicalGroupBindings, $canonicalGroupNames, forgetCanonicalGroup, registerCanonicalGroup } = await import('./canonical-group-registry')
    const open = vi.fn((_id: string, _options: { title: string }) => () => undefined)
    host.openWorkspace = open
    const firstRoute = { connectionId: 'first-owner', profile: 'team' }
    const secondRoute = { connectionId: 'second-owner', profile: 'team' }
    const first = registerCanonicalGroup(firstRoute, { room_id: 'same-room', name: 'Planning', members: [] })
    const second = registerCanonicalGroup(secondRoute, { room_id: 'same-room', name: 'Review', members: [] })
    const originalBinding = $canonicalGroupBindings.get()[first]

    room.view.openGroupChat(first)
    room.view.openGroupChat(second)
    expect(open.mock.calls.map(call => (call[1] as { title: string }).title)).toEqual(['Planning', 'Review'])
    expect(originalBinding).toEqual({ ...firstRoute, roomId: 'same-room' })

    expect(registerCanonicalGroup(firstRoute, { room_id: 'same-room', name: 'Renamed planning', members: [] })).toBe(first)
    room.view.openGroupChat(first)
    expect((open.mock.calls.at(-1)?.[1] as { title: string }).title).toBe('Renamed planning')
    expect($canonicalGroupBindings.get()[first]).toBe(originalBinding)

    forgetCanonicalGroup(originalBinding)
    expect($canonicalGroupNames.get()[first]).toBeUndefined()
    expect($canonicalGroupNames.get()[second]).toBe('Review')
  })
})

describe('disband', () => {
  it.each([false, true])('deferred creation cannot restore membership after disband (replacement: %s)', async replace => {
    const room = await loadRoom()
    const { CreateGroupChatDialog } = await import('./create-dialog')
    const request = host.request as (method: string, params: Record<string, unknown>) => Promise<unknown>
    let finish!: (value: unknown) => void
    const pending = new Promise(resolve => { finish = resolve })
    let held = false

    host.request = (method: string, params: Record<string, unknown>) => {
      if (method === 'profiles.configure' && params.name === 'research' && !held) {
        held = true

        return pending
      }

      return request(method, params)
    }

    const roster = [{ name: 'research' }, { name: 'builder' }]
    const onCreated = vi.fn()
    render(createElement(CreateGroupChatDialog, { onClose: vi.fn(), onCreated, open: true, roster }))
    screen.getAllByRole('checkbox').forEach(box => fireEvent.click(box))
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: CANONICAL_GROUP_LOCALES.en.createGroup })) })
    const group = Object.keys(room.chat.$groupChats.get())[0]
    await act(async () => { await room.view.disbandGroupChat(group, roster) })
    expect(room.chat.$groupChats.get()[group]).toBeUndefined()
    expect(room.data.$botMeta.get().builder.groups).not.toContain(group)

    if (replace) {room.chat.updateGroupChat(group, current => ({ ...current, roomId: 'replacement' }), { sync: false })}
    const metadata = structuredClone(room.data.$botMeta.get())
    await act(async () => { finish({ applied: { ui_meta: true } }) })
    expect(room.data.$botMeta.get()).toEqual(metadata)
    expect(room.chat.$groupChats.get()[group]?.roomId).toBe(replace ? 'replacement' : undefined)
    expect(onCreated).not.toHaveBeenCalled()
  })

  it('cold reload keeps a disbanded room gone and every other room field intact', async () => {
    const room = await loadRoom()
    room.chat.updateGroupChat('Keep', current => ({
      ...current,
      log: [{ at: 10, from: { kind: 'user', name: 'You' }, id: 'held', text: 'keep', thread: 't' }],
      watermarks: { 't::builder': 1 }, sessions: { builder: 'session' },
      sessionOwners: { builder: { name: 'builder', connectionId: 'owner' } },
      holds: { builder: { at: 2, noted: true } }, heldMessages: { builder: ['held'] },
      holdDetection: false, stranded: { builder: { before: 2, thread: 't' } },
      externalCursors: { builder: 4 }, members: [{ name: 'builder' }],
      roomId: 'keep-id', image: 'data:image/png;base64,keep', sectionId: 'section',
      rosterOrder: 3, pinned: true, syncRevision: 7
    }), { sync: false })
    room.chat.updateGroupChat('Gone', current => ({ ...current, roomId: 'gone-id', running: true }), { sync: false })
    const keep = structuredClone(durable(room).Keep)
    await room.view.disbandGroupChat('Gone', [])
    const stored = JSON.parse(JSON.stringify(durable(room)))
    expect(stored).toEqual({ Keep: keep })
    // Replace all runtime state with the cold-start reader, not the live atom.
    room.chat.$groupChats.set(room.chat.hydrateGroupChatRooms(stored))
    expect(room.chat.$groupChats.get().Gone).toBeUndefined()
    expect(room.chat.durableGroupChatRooms()).toEqual({ Keep: keep })
    room.chat.updateGroupChat('Keep', current => current, { sync: false })
    expect(durable(room)).toEqual({ Keep: keep })
  })
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

  it('outlives a gateway mirror that missed the tombstone push (#105275)', async () => {
    const room = await loadRoom()
    room.chat.$groupChats.set({
      Build: {
        log: [{ at: 1, from: { kind: 'user', name: 'You' }, id: 'b1', text: 'go' }],
        roomId: 'room-1',
        syncRevision: 4,
        watermarks: {}
      }
    } as unknown as Record<string, GroupChat>)

    await room.view.disbandGroupChat('Build', [])

    // The disband is remembered durably, not only in the in-flight sync job.
    expect(room.gateway.storage.get('group-chat-tombstones')).toEqual({ 'id:room-1': 5 })

    // A lagging mirror (its tombstone push failed) still projects the room
    // with a higher CAS revision and no tombstone. Pulling it must not
    // resurrect the room — and the durable room map must stay clean.
    room.gateway.uiMeta['hermes-bots-groups'] = {
      rooms: {
        'id:room-1': {
          log: [{ at: 1, from: { kind: 'user', name: 'You' }, id: 'b1', text: 'go' }],
          members: [],
          name: 'Build',
          revision: 9,
          roomId: 'room-1'
        }
      },
      updatedAt: 2,
      version: 3
    }

    await room.chat.pullGroupChatServerState()

    expect('Build' in room.chat.$groupChats.get()).toBe(false)
    expect('Build' in durable(room)).toBe(false)
  })
})

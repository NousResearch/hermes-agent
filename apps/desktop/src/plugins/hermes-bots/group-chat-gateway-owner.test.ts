import { expect, it, vi } from 'vitest'
/** Gateway execution ownership survives local storage and room edits. */

import type * as GroupChatModule from './group-chat'
import { scriptedStorage } from './group-test-utils'
import type * as I18nModule from './i18n'
import type { RosterRow } from './types'

const { onEvent, request } = vi.hoisted(() => ({
  onEvent: vi.fn(() => () => undefined),
  request: vi.fn(async () => ({}))
}))

vi.mock('@hermes/plugin-sdk', async () => {
  const { atom } = await import('nanostores')

  const stub: unknown = new Proxy(function stubbed() {}, {
    apply: () => stub,
    get: (_target, key) => (key === 'then' ? undefined : stub),
    has: () => true
  })

  const hostMock = {
    onEvent,
    paneVisibility: vi.fn(() => ({ get: () => false, listen: () => () => undefined })),
    request,
    state: {
      focusedStoredSessionId: { get: () => null, listen: () => () => undefined },
      gateway: { get: () => 'open', listen: () => () => undefined },
      connectionId: { get: () => 'owner', listen: () => () => undefined },
      profile: { get: () => 'default', listen: () => () => undefined }
    }
  }

  const known: Record<string, unknown> = { atom, host: hostMock }

  return new Proxy(known, {
    get: (target, key) =>
      typeof key === 'symbol' || key in target ? target[key as string] : key === 'then' ? undefined : stub,
    has: () => true
  })
})

vi.mock('./avatar', () => ({ startFaceClock: vi.fn(), stopFaceClock: vi.fn() }))
vi.mock('./canonical-chat', () => ({
  CANONICAL_CHAT_TITLE: 'Bot Chat',
  isCanonicalChatOnScreen: vi.fn(() => false),
  notifyBotOpenFailure: vi.fn(),
  openBotCanonicalChat: vi.fn(async () => ({ openedId: 'bot-chat-tip', registryId: 'bot-chat' }))
}))
vi.mock('./chat-empty', () => ({ BotChatEmpty: () => null }))
vi.mock('./relay', () => ({ startBotRelay: vi.fn(), stopBotRelay: vi.fn() }))
vi.mock('./session-sweep', () => ({ startHideSweepScheduler: vi.fn() }))
vi.mock('./cron', () => ({ bindProfileSync: () => () => undefined, RoutinesPane: () => null }))
vi.mock('./screen-autoraise', () => ({ startScreenAutoRaise: vi.fn(() => () => undefined) }))
vi.mock('./hygiene', () => ({ annotateOrphanedGroupChatMembers: () => ({ changed: false, rooms: {} }) }))
vi.mock('./group-chat', async importOriginal => ({
  ...(await importOriginal<typeof GroupChatModule>()),
  pullGroupChatServerState: async () => false,
  scheduleGroupChatServerSync: vi.fn()
}))
vi.mock('./user-sections', () => ({ loadBotSections: vi.fn() }))
vi.mock('./data', async () => {
  const { atom: nanoAtom } = await import('nanostores')

  return {
    ROSTER_KEY: ['hermes-bots'],
    $botMeta: nanoAtom({}),
    $lastRoster: nanoAtom<RosterRow[]>([]),
    botHandle: (name: string) => name,
    botMentionTag: (name: string) => name,
    botRosterKey: (bot: { connectionId?: string; name?: string } | null | undefined) =>
      `${bot?.connectionId || 'legacy'}::${bot?.name || 'default'}`,
    botSelectionKey: (bot: RosterRow) => bot.name,
    cachedUnionRoster: () => null,
    isActiveRosterBot: () => false,
    migrateBotMeta: vi.fn(async () => undefined),
    primeRoster: vi.fn(async () => undefined),
    resolveRosterMentions: vi.fn(() => ({}))
  }
})
vi.mock('./i18n', async importOriginal => {
  const actual = await importOriginal<typeof I18nModule>()

  return { ...actual, BOTS_LOCALES: {} }
})
const plugin = (await import('./plugin')).default

const {
  $groupChats,
  updateGroupChat,
  durableGroupChatRooms,
  groupChatSyncSnapshot,
  mergeRemoteGroupChatSnapshotIntoRooms,
  setGroupChatSyncDisposed
} = await import('./group-chat')

const { setPluginCtx } = await import('./shared')
const { renameGroupChat } = await import('./group-chat-view')
const { registerCanonicalGroup } = await import('./canonical-group-registry')
const { setGroupChatMembers } = await import('./group-chat-view-members')
const { applyGroupHoldDirective } = await import('./group-rounds')
const { runGroupRoundMember } = await import('./group-round-members')

const engine = { kind: 'gateway', connectionId: 'owner', profile: 'default', roomId: 'r1' } as const

function hydrationContext(values: Map<string, unknown>) {
  return {
    ...scriptedStorage(values),
    i18n: { register: () => () => undefined },
    onDispose: () => undefined,
    register: () => () => undefined
  }
}

it('round trips the gateway owner through real writes, hydration, membership edits and rename', async () => {
  const values = new Map<string, unknown>()

  const ctx = hydrationContext(values)

  setGroupChatSyncDisposed(true)
  setPluginCtx(ctx as never)
  updateGroupChat(
    'Mixed',
    room => ({
      ...room,
      roomId: 'r1',
      engine: engine as never,
      members: [
        { name: 'alpha', connectionId: 'owner' },
        { name: 'default', connectionId: 'peer', remoteSource: true }
      ]
    }),
    { sync: false }
  )
  expect((values.get('group-chats') as ReturnType<typeof $groupChats.get>).Mixed.engine).toEqual(engine)
  $groupChats.set({})

  try {
    plugin.register(ctx as never)
  } catch {
    /* UI surfaces are outside this storage fixture. */
  }
  await vi.waitFor(() => expect($groupChats.get().Mixed?.engine).toEqual(engine))
  await setGroupChatMembers('Mixed', [
    { name: 'alpha', connectionId: 'owner' },
    { name: 'beta', connectionId: 'owner' }
  ])
  expect(durableGroupChatRooms().Mixed.engine).toEqual(engine)
  await renameGroupChat('Mixed', 'Renamed', [])
  expect((values.get('group-chats') as ReturnType<typeof $groupChats.get>).Renamed.engine).toEqual(engine)
  expect(groupChatSyncSnapshot().rooms).toEqual({})
  const holds = applyGroupHoldDirective({}, { everyone: false, mentioned: ['alpha'] }, '@alpha stop', { at: 1 })
  const sessionOwners = { alpha: { connectionId: 'owner', profile: 'alpha' } }
  updateGroupChat('Held Desktop', room => ({ ...room, engine: 'desktop', holds, sessionOwners }), { sync: false })
  registerCanonicalGroup(engine, { room_id: engine.roomId, name: 'Gateway rename', members: [] })
  expect((values.get('group-chats') as ReturnType<typeof $groupChats.get>)['Gateway rename'].engine).toEqual(engine)
  expect($groupChats.get().Renamed).toBeUndefined()
  $groupChats.set({})

  try {
    plugin.register(ctx as never)
  } catch {
    /* UI surfaces are outside this storage fixture. */
  }
  await vi.waitFor(() => expect($groupChats.get()['Gateway rename']?.engine).toEqual(engine))
  expect($groupChats.get()['Held Desktop'].holds).toEqual(holds)
  expect($groupChats.get()['Held Desktop'].sessionOwners).toEqual(sessionOwners)
  updateGroupChat(
    'Held Desktop',
    room => ({
      ...room,
      log: [{ id: 'after-restart', thread: 't1', at: 2, from: { kind: 'user', name: 'You' }, text: 'Any updates?' }]
    }),
    { sync: false }
  )
  await runGroupRoundMember(
    {
      group: 'Held Desktop',
      members: [{ name: 'alpha' }],
      thread: 't1',
      startEpoch: 0,
      binding: { isLive: () => true },
      isCurrent: () => true
    },
    { name: 'alpha' }
  )
  expect($groupChats.get()['Held Desktop'].heldMessages?.alpha).toEqual(['after-restart'])
  setGroupChatSyncDisposed(true)
  setPluginCtx(null)
})

it.each([false, true])(
  "keeps another Desktop's same-name room separate from the saved gateway owner (rename: %s)",
  renamed => {
    const local = { Team: { roomId: engine.roomId, engine, log: [], watermarks: {} } }

    const remote = {
      Team: {
        roomId: 'another-room',
        syncRevision: 1,
        engine: 'desktop' as const,
        watermarks: {},
        members: [{ name: 'beta', connectionId: 'another-desktop' }],
        log: [{ id: 'remote-message', at: 1, from: { kind: 'user' as const, name: 'You' }, text: 'Other conversation' }]
      }
    }

    const merged = mergeRemoteGroupChatSnapshotIntoRooms(groupChatSyncSnapshot(remote), {
      ...local,
      ...(renamed ? { 'Earlier name': { ...remote.Team, syncRevision: 0 } } : {})
    })

    expect(merged.Team).toEqual(local.Team)
  }
)

it('keeps gateway ownership through a real rename mirror tombstone and storage reload', async () => {
  const values = new Map<string, unknown>()

  const ctx = hydrationContext(values)

  const { mergeGroupChatSyncSnapshots, pullGroupChatServerState } =
    await vi.importActual<typeof GroupChatModule>('./group-chat')

  const desktop = {
    engine: 'desktop' as const,
    watermarks: {},
    log: [{ id: 'old-message', at: 1, from: { kind: 'user' as const, name: 'You' }, text: 'Desktop history' }]
  }

  const mirror = mergeGroupChatSyncSnapshots(
    groupChatSyncSnapshot({ Team: desktop, Ordinary: desktop }),
    groupChatSyncSnapshot({ Renamed: desktop }),
    { changedRooms: ['Renamed'], deletedRooms: ['Team', 'Ordinary'], writeRevision: 2 }
  )

  expect(mirror.deleted).toMatchObject({ 'name:Team': 2, 'name:Ordinary': 2 })
  setGroupChatSyncDisposed(true)
  setPluginCtx(ctx as never)
  $groupChats.set({})
  updateGroupChat('Team', room => ({ ...room, roomId: engine.roomId, engine }), { sync: false })
  updateGroupChat('Ordinary', () => desktop, { sync: false })
  request.mockResolvedValue({ profiles: [{ name: 'default', ui_meta: { 'hermes-bots-groups': mirror } }] })
  await pullGroupChatServerState('owner')
  $groupChats.set({})

  try {
    plugin.register(ctx as never)
  } catch {
    /* UI surfaces are outside this storage fixture. */
  }
  await vi.waitFor(() => expect($groupChats.get().Renamed?.log).toHaveLength(1))
  expect($groupChats.get().Team?.engine).toEqual(engine)
  expect($groupChats.get().Ordinary).toBeUndefined()
  setGroupChatSyncDisposed(true)
  setPluginCtx(null)
})

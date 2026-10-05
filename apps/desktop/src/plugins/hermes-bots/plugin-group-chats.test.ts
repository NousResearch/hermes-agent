import type { PluginContext } from '@hermes/plugin-sdk'
import { host } from '@hermes/plugin-sdk'
import { expect, it, vi } from 'vitest'
const groupChats = host.groupChats

vi.mock('./avatar', () => ({ startFaceClock: vi.fn(), stopFaceClock: vi.fn() }))
vi.mock('./relay', () => ({ startBotRelay: vi.fn(), stopBotRelay: vi.fn() }))
vi.mock('./screen-autoraise', () => ({ startScreenAutoRaise: () => () => {} }))
vi.mock('./session-sweep', () => ({ startHideSweepScheduler: vi.fn() }))
vi.mock('./canonical-chat', () => ({ openBotCanonicalChat: vi.fn() }))
vi.mock('./chat-empty', () => ({ BotChatEmpty: () => null }))
vi.mock('./cron', () => ({ bindProfileSync: () => () => {}, RoutinesPane: () => null }))
vi.mock('./roster-pane', () => ({
  botChatOwnsWorkspace: () => false,
  BotsPane: () => null,
  releaseStaleOpenBotChat: vi.fn(),
  selectedRosterBot: () => null,
  sessionOwnsWorkspace: () => false
}))
vi.mock('./data', async original => ({
  ...(await original<object>()),
  migrateBotMeta: async () => {},
  primeRoster: async () => {}
}))
vi.mock('./group-chat', async original => ({
  ...(await original<object>()),
  pullGroupChatServerState: async () => false,
  scheduleGroupChatServerSync: vi.fn()
}))

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(done => {
    resolve = done
  })

  return { promise, resolve }
}

function context(rooms: Promise<unknown>, tombstones: Promise<unknown>) {
  const disposers: (() => void)[] = []

  return {
    ctx: {
      i18n: { register: () => () => {}, t: (key: string) => key },
      onDispose: (fn: () => void) => disposers.push(fn),
      register: () => () => {},
      storage: {
        get: (key: string) =>
          key === 'group-chats' ? rooms : key === 'group-chat-tombstones' ? tombstones : Promise.resolve(undefined),
        set: async () => {}
      }
    } as unknown as PluginContext,
    dispose: () => disposers.forEach(fn => fn())
  }
}

const settle = () => new Promise(resolve => setTimeout(resolve, 0))

const plugin = (await import('./plugin')).default

it.each(['rooms', 'tombstones'])('fails closed when %s hydration cannot be read', async failed => {
  const rooms = deferred<unknown>()

  const current =
    failed === 'rooms' ? context(rooms.promise, Promise.resolve({})) : context(Promise.resolve({}), rooms.promise)

  plugin.register(current.ctx)

  try {
    // A storage getter may reject before either gateway or views exist.
    const chat = await import('./group-chat')
    vi.mocked(chat.scheduleGroupChatServerSync).mockClear()
    rooms.resolve(Promise.reject(new Error('storage unavailable')))
    await settle()
    expect(groupChats.status()).toBe('unavailable')
    expect(chat.scheduleGroupChatServerSync).not.toHaveBeenCalled()
  } finally {
    current.dispose()
  }
})

it('registers independently of views, awaits both hydration inputs and ignores disposed-generation hydration', async () => {
  const chat = await import('./group-chat')
  const oldRooms = deferred<unknown>()
  const oldTombstones = deferred<unknown>()
  const old = context(oldRooms.promise, oldTombstones.promise)
  plugin.register(old.ctx)
  expect(groupChats.status()).toBe('loading')
  old.dispose()
  expect(groupChats.status()).toBe('unavailable')
  const rooms = deferred<unknown>()
  const tombstones = deferred<unknown>()
  const current = context(rooms.promise, tombstones.promise)
  plugin.register(current.ctx)

  try {
    rooms.resolve({ Current: { roomId: 'current', log: [], watermarks: {} } })
    await settle()
    expect(groupChats.status()).toBe('loading')
    tombstones.resolve({})
    await settle()
    expect(groupChats.status()).toBe('ready')
    oldRooms.resolve({ Stale: { roomId: 'stale', log: [], watermarks: {} } })
    oldTombstones.resolve({ 'id:current': 100 })
    await settle()
    expect(groupChats.getSnapshot().rooms.map(room => room.roomId)).toContain('current')
    expect(chat.$groupChats.get()).not.toHaveProperty('Stale')
    expect(chat.groupChatTombstoneMemory()).not.toHaveProperty('id:current')
    expect(groupChats.status()).toBe('ready')
  } finally {
    current.dispose()
  }

  expect(groupChats.status()).toBe('unavailable')
})

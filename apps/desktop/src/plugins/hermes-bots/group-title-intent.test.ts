import { afterEach, expect, it, vi } from 'vitest'

import { createGroupGateway, drain, scriptedStorage } from './group-test-utils'
import type { GroupChat } from './types'

const { host } = vi.hoisted(() => ({ host: {} as Record<string, unknown> }))

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')

  return pluginSdkMock(host)
})

const key = 'hermes-bots-groups'

function room(title = 'Old title', revision = 1): GroupChat {
  return {
    log: [{ id: 'root', thread: 'topic', at: 1, from: { kind: 'user', name: 'You' }, text: 'Root' }],
    watermarks: {}, threadTitles: { topic: title }, syncRevision: revision
  }
}

/** Fire only the chosen debounce/backoff; async RPC continuations drain separately. */
function controlledTimers() {
  let next = 0
  const pending = new Map<number, { callback: () => void; delay: number }>()

  vi.stubGlobal('setTimeout', (callback: () => void, delay: number) => {
    pending.set(++next, { callback, delay })

    return next
  })
  vi.stubGlobal('clearTimeout', (id: number) => pending.delete(id))

  return async (delay = 350) => {
    const entry = [...pending].find(([, timer]) => timer.delay === delay)

    expect(entry, `expected a ${delay}ms timer`).toBeDefined()
    pending.delete(entry![0])
    entry![1].callback()
    await drain(() => true, 12)
  }
}

async function setup() {
  vi.resetModules()
  const gateway = createGroupGateway()

  for (const name of Object.keys(host)) { delete host[name] }
  Object.assign(host, gateway.host)
  const chat = await import('./group-chat')
  const { setPluginCtx } = await import('./shared')

  setPluginCtx(scriptedStorage(gateway.storage))

  return { chat, gateway, fire: controlledTimers() }
}

afterEach(() => vi.unstubAllGlobals())

it('keeps a newer title edit before secondary fan-out when an older secondary ack arrives', async () => {
  const { chat, gateway, fire } = await setup()
  const secondary = createGroupGateway()
  const localRequest = gateway.host.request as (method: string, params: Record<string, unknown>) => Promise<unknown>
  const secondaryRequest = secondary.host.request as typeof localRequest

  let release = () => {}
  const delayed = new Promise<void>(resolve => { release = resolve })
  let secondaryWrites = 0

  host.profileRoutes = async () => [
    { connectionId: 'local', profile: 'default' },
    { connectionId: 'secondary', profile: 'default' }
  ]

  host.requestProfile = async (route: { connectionId: string }, method: string, params: Record<string, unknown>) => {
    if (route.connectionId === 'local') { return localRequest(method, params) }

    if (method === 'profiles.configure' && ++secondaryWrites === 1) { await delayed }

    return secondaryRequest(method, params)
  }

  for (const target of [gateway, secondary]) {
    target.uiMeta[key] = chat.groupChatSyncSnapshot({ Room: room() })
    target.uiMetaRevisions[key] = 1
  }

  chat.$groupChats.set({ Room: room() })
  chat.updateGroupChat('Room', value => ({ ...value, threadTitles: { topic: 'First edit' } }))
  await fire()
  expect(gateway.uiMetaRevisions[key]).toBe(2)
  expect(secondaryWrites).toBe(1)
  expect(secondary.uiMetaRevisions[key]).toBe(1)
  chat.updateGroupChat('Room', value => ({ ...value, threadTitles: { topic: 'Second edit' } }))
  expect(chat.$groupChats.get().Room.threadTitles?.topic).toBe('Second edit')
  release()
  await drain(() => true, 12)
  expect(secondary.uiMetaRevisions[key]).toBe(2)
  expect(chat.$groupChats.get().Room.threadTitles?.topic).toBe('Second edit')
  await fire()

  for (const target of [gateway, secondary]) {
    const published = target.uiMeta[key] as ReturnType<typeof chat.groupChatSyncSnapshot>

    expect(published.rooms['name:Room'].threadTitles?.topic).toBe('Second edit')
    expect(target.uiMetaRevisions[key]).toBe(3)
  }

  chat.stopGroupChatServerSync()
})

it.each([false, true])('does not replay acknowledged title intent after failed read-back (new edit: %s)', async newerEdit => {
  const { chat, gateway, fire } = await setup()
  const request = gateway.host.request as (method: string, params: Record<string, unknown>) => Promise<unknown>
  let failReadBack = false
  let configured = 0
  let failedReads = 0

  host.request = async (method: string, params: Record<string, unknown> = {}) => {
    if (method === 'profiles.list' && failReadBack) {
      failReadBack = false
      failedReads++
      throw new Error('read-back unavailable after acknowledged configure')
    }

    const result = await request(method, params)

    if (method === 'profiles.configure' && ++configured === 1) { failReadBack = true }

    return result
  }

  gateway.uiMeta[key] = chat.groupChatSyncSnapshot({ Room: room() })
  gateway.uiMetaRevisions[key] = 1
  chat.$groupChats.set({ Room: room() })
  chat.updateGroupChat('Room', value => ({ ...value, threadTitles: { topic: 'First edit' } }))
  await fire()
  expect(gateway.uiMetaRevisions[key]).toBe(2)
  expect(failedReads).toBe(1)
  gateway.uiMeta[key] = chat.groupChatSyncSnapshot({ Room: room('', 3) })
  gateway.uiMetaRevisions[key] = 3

  if (newerEdit) {
    chat.updateGroupChat('Room', value => ({ ...value, threadTitles: { topic: 'Second edit' } }))
  }

  await fire(newerEdit ? 350 : 2000)
  const published = gateway.uiMeta[key] as ReturnType<typeof chat.groupChatSyncSnapshot>

  expect(configured).toBe(2)
  expect(published.rooms['name:Room'].threadTitles?.topic).toBe(newerEdit ? 'Second edit' : '')
  expect(chat.$groupChats.get().Room.threadTitles?.topic).toBe(newerEdit ? 'Second edit' : '')
  chat.stopGroupChatServerSync()
})

it.each(['stable identity', 'explicit reset', 'identity assigned during debounce'])(
  'publishes title intent with %s', async scenario => {
    const { chat, gateway, fire } = await setup()
    const local = room()
    const desired = scenario === 'explicit reset' ? '' : 'Local title'

    if (scenario !== 'identity assigned during debounce') { local.roomId = 'abc' }
    const remote = chat.groupChatSyncSnapshot({ Room: { ...room('Remote title', 9), roomId: 'abc' } })

    gateway.uiMeta[key] = remote
    gateway.uiMetaRevisions[key] = 9
    chat.$groupChats.set({ Room: local })
    chat.updateGroupChat('Room', value => ({ ...value, threadTitles: { topic: desired } }))

    if (scenario === 'identity assigned during debounce') {
      chat.$groupChats.set(chat.mergeRemoteGroupChatSnapshotIntoRooms(remote, chat.$groupChats.get(), { preserveRooms: ['Room'] }))
      expect(chat.$groupChats.get().Room.roomId).toBe('abc')
    }

    await fire()
    const published = gateway.uiMeta[key] as ReturnType<typeof chat.groupChatSyncSnapshot>

    expect(gateway.rpcFor('profiles.configure')).toHaveLength(1)
    expect(published.rooms['id:abc'].threadTitles?.topic).toBe(desired)
    chat.stopGroupChatServerSync()
  }
)

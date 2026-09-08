import { beforeEach, expect, it, vi } from 'vitest'

import { createGroupGateway, runTimersInline, scriptedStorage } from './group-test-utils'

const { host } = vi.hoisted(() => ({ host: {} as Record<string, unknown> }))

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')

  return pluginSdkMock(host)
})

beforeEach(() => {
  vi.resetModules()
  runTimersInline()

  for (const key of Object.keys(host)) {
    delete host[key]
  }
})

it.each(['stable-room-id', undefined])('keeps an in-flight reply on one room with identity %s', async roomId => {
  let release: (value: string) => void = () => undefined
  let entered: () => void = () => undefined

  const started = new Promise<void>(resolve => {
    entered = resolve
  })

  const reply = new Promise<string>(resolve => {
    release = resolve
  })

  const gateway = createGroupGateway({
    turn: ({ n }) => {
      if (n === 1) {
        entered()

        return reply
      }

      return '(pass)'
    }
  })

  Object.assign(host, gateway.host)
  const chat = await import('./group-chat')
  const rounds = await import('./group-rounds')
  const view = await import('./group-chat-view')
  const shared = await import('./shared')
  shared.setPluginCtx(scriptedStorage(gateway.storage))
  chat.updateGroupChat('Old', room => ({
    ...room,
    roomId,
    running: true,
    epoch: 1,
    log: [{ id: 'user-entry', at: 1, from: { kind: 'user', name: 'You' }, text: 'Continue', thread: 'thread-1' }]
  }))
  const drive = rounds.runGroupChatRounds('Old', [{ name: 'research', title: 'Research' }], 'thread-1')
  await started
  let renamed: null | string = null

  try {
    renamed = await view.renameGroupChat('Old', 'New', [])
  } finally {
    release('Completed reply')
    await drive
  }

  const current = chat.$groupChats.get()
  console.info('RENAME_RECEIPT', {
    roomId,
    renamed,
    names: Object.keys(current),
    replies: Object.fromEntries(
      Object.entries(current).map(([name, room]) => [
        name,
        room.log.filter(entry => entry.from.kind === 'member').map(entry => entry.text)
      ])
    )
  })
  expect(renamed).toBeNull()
  expect(Object.keys(current)).toEqual(['Old'])
  expect(current.Old.log.filter(entry => entry.from.kind === 'member').map(entry => entry.text)).toEqual([
    'Completed reply'
  ])
  expect(current.Old.running).toBe(false)
  expect(await view.renameGroupChat('Old', 'New', [])).toBe('New')
  expect(Object.keys(chat.$groupChats.get())).toEqual(['New'])
  expect(chat.$groupChats.get().New.roomId).toBe(roomId)
})

import { expect, it, vi } from 'vitest'

import { loadRuntimePlugin, unloadRuntimePlugin } from '@/contrib/runtime-loader'

import { registerGroupChatsProvider } from './group-chats'

it('loads a public-only consumer, preserves refused drafts, and disposes its subscription', async () => {
  const realBlob = globalThis.Blob
  vi.stubGlobal(
    'Blob',
    class {
      constructor(public parts: string[]) {}
    }
  )

  const create = vi
    .spyOn(URL, 'createObjectURL')
    .mockImplementation(
      blob =>
        `data:text/javascript;base64,${Buffer.from((blob as unknown as { parts: string[] }).parts.join('')).toString('base64')}`
    )

  const revoke = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {})
  const state = { draft: 'hello', refreshes: 0, submit: (_roomId: string) => {} }
  vi.stubGlobal('__groupChatsConsumer', state)

  let changed = () => {}
  const send = vi.fn(() => ({ accepted: true as const, threadId: 'thread' }))

  let unregister = () => {}

  try {
    const id = await loadRuntimePlugin(
      `
      import { host } from '@hermes/plugin-sdk'
      export default {
        id: 'group-chats-contract',
        register(ctx) {
          const consumer = globalThis.__groupChatsConsumer
          const refresh = () => { host.groupChats.getSnapshot(); consumer.refreshes++ }
          ctx.onDispose(host.groupChats.subscribe(refresh))
          consumer.submit = roomId => {
            const result = host.groupChats.send({ roomId, text: consumer.draft })
            if (result.accepted) consumer.draft = ''
          }
          refresh()
        }
      }
    `,
      'group-chats-contract-fixture'
    )

    expect(id).toBe('group-chats-contract')
    state.submit('room')
    expect(state.draft).toBe('hello')
    unregister = registerGroupChatsProvider({
      status: () => 'ready',
      getSnapshot: () => ({ rooms: [] }),
      subscribe: listener => {
        changed = listener

        return () => {}
      },
      send
    })
    state.submit('room')
    expect(state.draft).toBe('')
    expect(send).toHaveBeenCalledExactlyOnceWith({ roomId: 'room', text: 'hello' })
    unloadRuntimePlugin(id!)
    const count = state.refreshes
    changed()
    expect(state.refreshes).toBe(count)
  } finally {
    unloadRuntimePlugin('group-chats-contract')
    unregister()
    create.mockRestore()
    revoke.mockRestore()
    vi.stubGlobal('Blob', realBlob)
    vi.unstubAllGlobals()
  }
})

import { expect, it, vi } from 'vitest'

import { createGroupGateway, scriptedStorage } from './group-test-utils'
import { hostedRoomKey } from './hosted-room-protocol'

const { host } = vi.hoisted(() => ({ host: {} as Record<string, unknown> }))
vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')

  return pluginSdkMock(host)
})

it('rejects hosted identities at every legacy coordinator/session entry without local state or RPC writes', async () => {
  vi.resetModules()
  // Existing in-memory gateway helper, not a live backend or agent run.
  const gateway = createGroupGateway()
  Object.assign(host, gateway.host)

  const [chat, rounds, turns, shared] = await Promise.all([
    import('./group-chat'), import('./group-rounds'), import('./group-turns'), import('./shared')
  ])

  shared.setPluginCtx(scriptedStorage(gateway.storage))
  const key = hostedRoomKey({ connectionId: 'mock-owner', authorityGatewayId: 'mock-authority', roomId: 'mock-room' })
  const member = { name: 'mock-member', title: '' }
  const before = chat.$groupChats.get()

  expect(() => rounds.sendToGroupChat(key, [member], 'Mock input')).toThrow(/groups.send/)
  await expect(rounds.runGroupChatRounds(key, [member], 'mock-thread')).rejects.toThrow(/Desktop coordinator/)
  await expect(rounds.stopGroupThread(key, 'mock-thread', [member])).rejects.toThrow(/groups.stop/)
  await expect(turns.ensureGroupChatSession(key, member)).rejects.toThrow(/Desktop member sessions/)

  expect(chat.$groupChats.get()).toBe(before)
  expect(gateway.rpc).toEqual([])
  expect(gateway.calls).toEqual([])
  expect(gateway.storage.size).toBe(0)
  expect(gateway.sessions.size).toBe(0)
})

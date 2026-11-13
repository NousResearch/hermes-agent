import { beforeEach, expect, it, vi } from 'vitest'

import type * as groupChat from './group-chat'
import { createGroupGateway, runTimersInline, scriptedStorage } from './group-test-utils'
import type * as groupTurns from './group-turns'
import type { GroupMember } from './types'

// A Desktop-driven room seats members whose sessions live on the gateway that
// owns local sessions (gateway/session_controls.py). That gateway answers a
// missing session with a 4001 refusal whose reason is `not_found`, where a
// `hermes serve` socket answers 4007. Both mean "nothing to resume, mint one";
// every other refusal still fails closed.

const { host } = vi.hoisted(() => ({ host: {} as Record<string, unknown> }))

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')

  return pluginSdkMock(host)
})

const member: GroupMember = { name: 'research', title: '' }

async function loadRoom(refusal: { code: number; message: string; reason: string }) {
  vi.resetModules()
  const gateway = createGroupGateway()

  for (const key of Object.keys(host)) {
    delete host[key]
  }

  Object.assign(host, gateway.host)
  const request = host.request as (method: string, params: Record<string, unknown>) => Promise<unknown>

  // The owning gateway's answer to a resume it cannot find, in its own wire shape.
  host.request = async (method: string, params: Record<string, unknown>) => {
    if (method === 'session.resume' && !gateway.sessions.has(String(params.session_id))) {
      throw Object.assign(new Error(refusal.message), { code: refusal.code, data: { reason: refusal.reason } })
    }

    return request(method, params)
  }

  const [chat, turns, shared] = await Promise.all([import('./group-chat'), import('./group-turns'), import('./shared')])
  shared.setPluginCtx(scriptedStorage(gateway.storage))
  ;(chat as typeof groupChat).updateGroupChat('Mixed', current => {
    current.roomId = 'r-mixed'

    return current
  })

  return { chat: chat as typeof groupChat, gateway, turns: turns as typeof groupTurns }
}

beforeEach(() => {
  runTimersInline()
})

it("mints a member's first session when the owning gateway answers not_found", async () => {
  const room = await loadRoom({ code: 4001, message: 'not_found', reason: 'not_found' })

  const handle = await room.turns.ensureGroupChatSession('Mixed', member, 't1')

  expect(handle.fresh).toBe(true)
  expect(room.gateway.sessions.get(String(handle.stored))?.title).toBe('Group: r-mixed · t1')
  // The next turn resumes the stored session instead of minting another.
  expect((await room.turns.ensureGroupChatSession('Mixed', member, 't1')).stored).toBe(handle.stored)
  expect(room.gateway.sessions.size).toBe(1)
})

it('still fails closed on any other refusal from the owning gateway', async () => {
  const room = await loadRoom({ code: 4001, message: 'session_busy', reason: 'session_busy' })

  await expect(room.turns.ensureGroupChatSession('Mixed', member, 't1')).rejects.toThrow(/Could not check .*group session/)
  expect(room.gateway.sessions.size).toBe(0)
})

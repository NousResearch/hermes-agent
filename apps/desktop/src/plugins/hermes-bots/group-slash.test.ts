import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { createGroupGateway, drain, runTimersInline, scriptedStorage } from './group-test-utils'
import type { Attachment, GroupMember } from './types'

const { host } = vi.hoisted(() => ({ host: {} as Record<string, unknown> }))

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')

  return pluginSdkMock(host)
})

const MEMBERS: GroupMember[] = [
  { name: 'research', title: '' },
  { name: 'builder', title: '' }
]

const IMAGE: Attachment = { data: 'data:image/png;base64,iVBORw0KGgo=', kind: 'image', name: 'screenshot.png' }

async function loadRoom() {
  vi.resetModules()
  const gateway = createGroupGateway()
  Object.assign(host, gateway.host)

  const [chat, rounds, shared] = await Promise.all([
    import('./group-chat'),
    import('./group-rounds'),
    import('./shared')
  ])

  shared.setPluginCtx(scriptedStorage(gateway.storage))

  return { chat, rounds, gateway }
}

beforeEach(() => runTimersInline())
afterEach(() => vi.unstubAllGlobals())

it('rejects command-shaped sends visibly without changing the room or reaching a member, even with attachments', async () => {
  const { chat, rounds, gateway } = await loadRoom()
  const thread = rounds.sendToGroupChat('Team', MEMBERS, 'Ready')
  await drain(() => Boolean(chat.$groupChats.get().Team?.running))
  chat.$groupNeedsYou.set({ Team: true })

  const before = structuredClone(chat.$groupChats.get())
  const needsYou = chat.$groupNeedsYou.get()
  const storage = structuredClone(gateway.storage)
  gateway.rpc.length = 0
  gateway.calls.length = 0
  gateway.attaches.length = 0

  for (const text of ['/new', '  /ReSeT  ', '/custom-skill argument', '/custom_skill\nargument']) {
    for (const target of [null, thread]) {
      for (const attachments of [[], [IMAGE]]) {
        const draftAttachments = structuredClone(attachments)
        vi.mocked(host.notify as ReturnType<typeof vi.fn>).mockClear()

        expect(rounds.sendToGroupChat('Team', MEMBERS, text, target, attachments)).toBeNull()
        await drain(() => Boolean(chat.$groupChats.get().Team?.running))

        expect(host.notify).toHaveBeenCalledExactlyOnceWith({
          kind: 'warning',
          message: expect.stringMatching(/slash commands.*group chats/i)
        })
        expect(chat.$groupChats.get()).toEqual(before)
        expect(chat.$groupNeedsYou.get()).toBe(needsYou)
        expect(gateway.storage).toEqual(storage)
        expect(gateway.rpc).toEqual([])
        expect(gateway.calls).toEqual([])
        expect(gateway.attaches).toEqual([])
        expect(attachments).toEqual(draftAttachments)
      }
    }
  }
})

it('sends path-first text and embedded slash commands to the room members with attachments intact', async () => {
  const { chat, rounds, gateway } = await loadRoom()

  for (const text of ['/etc/hosts', '/tmp/report.txt inspect this', 'Explain /new', '/']) {
    gateway.calls.length = 0
    gateway.attaches.length = 0

    const thread = rounds.sendToGroupChat('Team', MEMBERS, text, null, [IMAGE])
    expect(thread).toEqual(expect.any(String))
    await drain(() => Boolean(chat.$groupChats.get().Team?.running))

    expect(chat.$groupChats.get().Team.log).toContainEqual(expect.objectContaining({ text, thread, images: [IMAGE] }))
    expect(gateway.calls.map(call => call.profile)).toEqual(MEMBERS.map(member => member.name))
    expect(gateway.calls.every(call => call.prompt.includes(text))).toBe(true)
    expect(gateway.attaches.map(call => call.data)).toEqual(MEMBERS.map(() => IMAGE.data))
    expect(host.notify).not.toHaveBeenCalled()
  }
})

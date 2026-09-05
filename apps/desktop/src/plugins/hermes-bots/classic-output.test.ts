import { webcrypto } from 'node:crypto'

import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { ClassicTurn } from './classic-output'
import { createGroupGateway, runTimersInline, scriptedStorage } from './group-test-utils'
import type { GroupMember } from './types'

const { host } = vi.hoisted(() => ({ host: {} as Record<string, any> }))
vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')

  return pluginSdkMock(host)
})

const members: GroupMember[] = [
  { name: 'writer', connectionId: 'producer', remoteSource: true },
  { name: 'reviewer', connectionId: 'consumer', remoteSource: true }
]

const bytes = new Uint8Array([0, 255, 10, 13, 240, 159, 145, 139])
const content = Buffer.from(bytes).toString('base64')
const digest = Buffer.from(await webcrypto.subtle.digest('SHA-256', bytes)).toString('hex')

const item = {
  artifact_id: 'rart_example',
  kind: 'file',
  name: 'welcome.bin',
  mime: 'application/octet-stream',
  size: bytes.length,
  sha256: digest
}

beforeEach(() => {
  vi.resetModules()
  runTimersInline()
  vi.stubGlobal('crypto', webcrypto)
  let now = Date.now()
  vi.spyOn(Date, 'now').mockImplementation(() => ++now)
})
afterEach(() => vi.restoreAllMocks())

async function setup(shareAt = 1) {
  const gateway = createGroupGateway({
    turn: ({ profile, n }) => (profile === 'writer' && n === 1 ? '@reviewer inspect the shared file' : '(pass)')
  })

  Object.keys(host).forEach(key => delete host[key])
  Object.assign(host, gateway.host)
  const original = host.requestProfile
  const exports = new Map<string, any>()
  let saved: { before: number; thread: string; classicTurn: ClassicTurn } | undefined
  const fault = { corrupt: false, offline: false, loseSubmit: false, unknown: false }
  const chat = await import('./group-chat')

  host.requestProfile = async (route: any, method: string, params: any) => {
    if (method === 'gateway.capabilities') {
      return { classic_output_export_v1: true, installation: route.connectionId }
    }

    if (method === 'session.export.read') {
      if (fault.offline) {
        throw new Error('offline')
      }

      const value = params.export_id
        ? [...exports.values()].find(row => row.export_id === params.export_id)
        : exports.get(params.request_id)

      if (!value || fault.unknown) {
        throw Object.assign(new Error('unknown'), { code: 4150 })
      }

      if (params.artifact_id) {
        return { ...value, item, content_base64: fault.corrupt ? 'AAAA' : content }
      }

      return value
    }

    if (method === 'prompt.submit' && params.classic_export) {
      const request = params.classic_export
      const index = exports.size + 1
      const first = index === shareAt

      const value = {
        export_id: `ce_${request.request_id}`,
        generation: 1,
        state: 'published',
        group_id: request.group_id,
        thread_id: request.thread_id,
        recipients: request.recipients,
        text:
          index < shareAt
            ? route.profile === 'writer'
              ? '@reviewer prepare'
              : '@writer please finish'
            : route.profile === 'writer' && first
              ? '@reviewer inspect the shared file'
              : '(pass)',
        items: route.profile === 'writer' && first ? [item] : []
      }

      exports.set(request.request_id, value)

      if (route.profile === 'writer' && first) {
        saved = structuredClone(chat.$groupChats.get().Workshop.stranded?.['producer::writer'] as typeof saved)
      }

      const result = await original(route, method, params)

      if (fault.loseSubmit) {
        fault.loseSubmit = false
        throw new Error('response lost')
      }

      return { ...result, classic_export: value }
    }

    return original(route, method, params)
  }

  const shared = await import('./shared')
  shared.setPluginCtx(scriptedStorage(gateway.storage))
  chat.updateGroupChat('Workshop', room => ({ ...room, roomId: 'workshop-id', members, continuityMode: 'desktop' }))
  const rounds = await import('./group-rounds')
  const turns = await import('./group-turns')
  const output = await import('./classic-output')

  return { gateway, chat, rounds, turns, output, fault, saved: () => saved }
}

it('natural writer output carries only refs, and the remote reviewer gets exact verified bytes', async () => {
  const room = await setup()
  // Continuity selection is already resolved; exercise the actual classic round driver.
  room.chat.appendGroupChatEntry(
    'Workshop',
    { kind: 'user', name: 'You' },
    '@writer create and share welcome.bin',
    'thread'
  )
  await room.rounds.runGroupChatRounds('Workshop', members, 'thread')
  const message = room.chat.$groupChats.get().Workshop.log.find(entry => entry.images?.[0]?.classicExport)
  expect(
    message,
    JSON.stringify({ room: room.chat.$groupChats.get().Workshop, calls: room.gateway.calls })
  ).toBeDefined()
  expect(message?.images?.[0].data).toBeUndefined()
  expect(message?.images?.[0].classicExport?.sha256).toBe(digest)
  const attached = room.gateway.attaches.find(call => call.profile === 'reviewer')
  expect(attached?.data).toBe(`data:${item.mime};base64,${content}`)
  expect(room.gateway.calls.find(call => call.profile === 'reviewer')?.prompt).toContain(
    '@file:attachments/welcome.bin'
  )
  const stored = JSON.stringify(room.gateway.storage.get('group-chats'))
  expect(stored).not.toContain(content)
  expect(stored).toContain('classicExport')
})

it('lost submit responses recover the accepted export without another writer submit', async () => {
  const room = await setup()
  room.fault.loseSubmit = true
  room.chat.appendGroupChatEntry('Workshop', { kind: 'user', name: 'You' }, 'create', 'thread')
  const reply = await room.turns.runGroupChatMemberTurn('Workshop', members[0], 'create and share', 'thread')
  expect(typeof reply === 'object' && reply?.images[0].classicExport).toBeTruthy()
  expect(room.gateway.calls).toHaveLength(1)
})

it('unknown submit outcomes stay pending and recover without re-execution', async () => {
  const room = await setup()
  room.fault.loseSubmit = true
  room.fault.unknown = true
  room.chat.appendGroupChatEntry('Workshop', { kind: 'user', name: 'You' }, 'create', 'thread')
  await expect(room.turns.runGroupChatMemberTurn('Workshop', members[0], 'create', 'thread')).rejects.toThrow('unknown')
  expect(room.chat.$groupChats.get().Workshop.stranded?.['producer::writer']).toBeDefined()
  room.fault.unknown = false
  await room.turns.harvestStrandedGroupReply('Workshop', members[0])
  expect(room.gateway.calls).toHaveLength(1)
  expect(room.chat.$groupChats.get().Workshop.log.some(entry => entry.images?.[0]?.classicExport)).toBe(true)
})

it('a continuation after the final ordinary round receives the newly exported bytes', async () => {
  const room = await setup(3)
  room.chat.appendGroupChatEntry('Workshop', { kind: 'user', name: 'You' }, '@writer create a file', 'thread')
  await room.rounds.runGroupChatRounds('Workshop', members, 'thread')
  const reviewer = room.gateway.calls.filter(call => call.profile === 'reviewer')
  expect(reviewer).toHaveLength(2)
  expect(reviewer[1].prompt).toContain('@file:attachments/welcome.bin')
  expect(room.gateway.attaches.find(call => call.profile === 'reviewer')?.data).toBe(
    `data:${item.mime};base64,${content}`
  )
})

it('published bytes survive deletion of the original producer session', async () => {
  const room = await setup()
  room.chat.appendGroupChatEntry('Workshop', { kind: 'user', name: 'You' }, 'create', 'thread')
  const reply = await room.turns.runGroupChatMemberTurn('Workshop', members[0], 'create', 'thread')

  if (!reply || typeof reply === 'string') {
    throw new Error('missing export')
  }

  const original = host.requestProfile

  host.requestProfile = async (route: any, method: string, params: any) => {
    if (method === 'session.resume' && params.session_id === reply.images[0].classicExport?.session) {
      throw Object.assign(new Error('session removed'), { code: 4007 })
    }

    return original(route, method, params)
  }

  const downloaded = await room.output.readClassicAttachment('Workshop', reply.images[0])
  expect(downloaded.data).toBe(`data:${item.mime};base64,${content}`)
  expect(room.gateway.calls).toHaveLength(1)
  expect(room.gateway.rpcFor('session.close')).toHaveLength(1)
})

it('an unrelated offline third member cannot block healthy classic text turns', async () => {
  const room = await setup()
  const roster = [...members, { name: 'offline', connectionId: 'offline', remoteSource: true }]
  room.chat.updateGroupChat('Workshop', current => ({ ...current, members: roster }))
  const original = host.requestProfile

  host.requestProfile = async (route: any, method: string, params: any) => {
    if (route.connectionId === 'offline') {
      throw new Error('unrelated source offline')
    }

    return original(route, method, params)
  }

  room.chat.appendGroupChatEntry('Workshop', { kind: 'user', name: 'You' }, 'discuss the welcome message', 'thread')
  await room.rounds.runGroupChatRounds('Workshop', roster, 'thread')
  expect(room.gateway.calls.some(call => call.profile === 'writer')).toBe(true)
  expect(room.gateway.calls.some(call => call.profile === 'reviewer')).toBe(true)
  expect(room.gateway.rpcFor('prompt.submit').every(call => call.params.classic_export === undefined)).toBe(true)
  expect(room.chat.$groupChats.get().Workshop.log.some(entry => entry.images?.some(image => image.classicExport))).toBe(
    false
  )
})

it('no-file classic disband adds no export capability prerequisite for offline members', async () => {
  const room = await setup()

  const request = vi.fn(async () => {
    throw new Error('offline')
  })

  host.requestProfile = request
  await room.output.retireClassicGroup(room.chat.$groupChats.get().Workshop)
  expect(request).not.toHaveBeenCalled()
  const view = await import('./group-chat-view')
  await view.disbandGroupChat('Workshop', [])
  expect(room.chat.$groupChats.get().Workshop).toBeUndefined()
  expect(request.mock.calls).not.toContainEqual(expect.arrayContaining(['gateway.capabilities']))
})

it('late harvest projects one deterministic ref and does not consume newer unseen peer messages', async () => {
  const room = await setup()
  room.chat.appendGroupChatEntry('Workshop', { kind: 'user', name: 'You' }, 'create', 'thread')
  await room.turns.runGroupChatMemberTurn('Workshop', members[0], 'create', 'thread')
  room.chat.appendGroupChatEntry('Workshop', { kind: 'member', name: 'reviewer' }, 'waiting', 'thread')
  room.chat.updateGroupChat('Workshop', current => ({ ...current, stranded: { 'producer::writer': room.saved()! } }))
  await room.turns.harvestStrandedGroupReply('Workshop', members[0])
  await room.turns.harvestStrandedGroupReply('Workshop', members[0])
  expect(room.chat.$groupChats.get().Workshop.log.filter(entry => entry.images?.[0]?.classicExport)).toHaveLength(1)
  expect(
    room.chat.$groupChats.get().Workshop.watermarks['thread::producer::writer'],
    JSON.stringify(room.chat.$groupChats.get().Workshop)
  ).toBe(1)
})

it.each(['stop', 'new-input', 'removed'] as const)('late output is withheld after %s', async change => {
  const room = await setup()
  room.chat.appendGroupChatEntry('Workshop', { kind: 'user', name: 'You' }, 'create', 'thread')
  await room.turns.runGroupChatMemberTurn('Workshop', members[0], 'create', 'thread')
  room.chat.updateGroupChat('Workshop', current => ({
    ...current,
    stranded: { 'producer::writer': room.saved()! },
    ...(change === 'stop' ? { holds: { 'producer::writer': {} } } : {}),
    ...(change === 'removed' ? { members: [members[1]] } : {})
  }))

  if (change === 'new-input') {
    room.chat.appendGroupChatEntry('Workshop', { kind: 'user', name: 'You' }, 'changed plan', 'thread')
  }

  await room.turns.harvestStrandedGroupReply('Workshop', members[0])
  expect(room.chat.$groupChats.get().Workshop.log.some(entry => entry.images?.[0]?.classicExport)).toBe(false)
})

it.each(['corrupt', 'offline', 'new-member', 'new-group'] as const)('refuses byte delivery for %s', async fault => {
  const room = await setup()
  room.chat.appendGroupChatEntry('Workshop', { kind: 'user', name: 'You' }, 'create', 'thread')
  const reply = await room.turns.runGroupChatMemberTurn('Workshop', members[0], 'create', 'thread')

  if (!reply || typeof reply === 'string') {
    throw new Error('missing export')
  }

  let recipient = members[1]

  if (fault === 'corrupt') {
    room.fault.corrupt = true
  }

  if (fault === 'offline') {
    room.fault.offline = true
  }

  if (fault === 'new-group') {
    room.chat.updateGroupChat('Workshop', current => ({ ...current, roomId: 'replacement' }))
  }

  if (fault === 'new-member') {
    recipient = { name: 'new-reviewer', connectionId: 'consumer', remoteSource: true }
    room.chat.updateGroupChat('Workshop', current => ({ ...current, members: [...members, recipient] }))
  }

  await expect(room.output.readClassicAttachment('Workshop', reply.images[0], recipient)).rejects.toThrow()
  expect(room.gateway.attaches).toHaveLength(0)
})

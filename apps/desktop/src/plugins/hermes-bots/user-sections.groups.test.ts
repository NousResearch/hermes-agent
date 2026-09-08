/**
 * Group chats file into user-made sections exactly like bots — with the one
 * structural difference the model encodes: a bot's membership is a field on
 * its profile meta (profile sync carries it), while a group's is a field on
 * its room record, because that record is the only durable identity a group
 * has. Filing must ride the room's own durable persistence path, never a
 * parallel list.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

const { saveBotMeta, storage } = vi.hoisted(() => ({
  saveBotMeta: vi.fn<(bot: { name: string }, patch: Record<string, unknown>) => Promise<unknown>>(),
  storage: new Map<string, unknown>()
}))

vi.mock('@hermes/plugin-sdk', async () => {
  const { atom } = await import('nanostores')

  return { atom, host: {} }
})

vi.mock('./data', async () => {
  const { atom } = await import('nanostores')
  const $botMeta = atom<Record<string, { sectionId?: null | string }>>({})
  const $lastRoster = atom<unknown[]>([])

  saveBotMeta.mockImplementation(async (bot: { name: string }, patch: Record<string, unknown>) => {
    $botMeta.set({ ...$botMeta.get(), [bot.name]: { ...$botMeta.get()[bot.name], ...patch } })

    return { serverOutcome: 'persisted', serverPersisted: true }
  })

  return {
    $botMeta,
    $lastRoster,
    botRosterKey: (bot: { connectionId?: string; name?: string }) => `${bot?.connectionId || 'legacy'}::${bot?.name || 'default'}`,
    saveBotMeta
  }
})

vi.mock('./routing', () => ({
  botRosterMeta: (bot: { name: string }, meta: Record<string, unknown>) => meta[bot.name]
}))

vi.mock('./shared', () => ({
  getPluginCtx: () => ({
    storage: {
      get: (key: string, fallback: unknown) => (storage.has(key) ? storage.get(key) : fallback),
      set: (key: string, value: unknown) => storage.set(key, value)
    }
  })
}))

// './group-chat' is deliberately NOT mocked: filing must go through the real
// room-store path (atom + durable plugin-storage record), with only its
// externals (sdk, data, routing, shared) faked above.
import { $botMeta } from './data'
import { $groupChats } from './group-chat'
import type { RosterRow } from './types'
import {
  $botSections,
  createBotSection,
  groupChatSectionId,
  groupRowsBySection,
  moveGroupChatsToSection,
  UNASSIGNED_SECTION_KEY
} from './user-sections'

const row = (name: string) => ({ bot: { name } as RosterRow, kind: 'bot' as const })
const groupRow = (name: string) => ({ kind: 'group' as const, name })

beforeEach(() => {
  storage.clear()
  $botMeta.set({})
  $botSections.set([])
  $groupChats.set({})
  saveBotMeta.mockClear()
})

describe('user sections — group chats', () => {
  it('filing a group writes sectionId on its room record and rides the durable group-chats persistence; clearing returns it to the bucket', () => {
    const section = createBotSection('XYZ')!

    moveGroupChatsToSection(['Team Chat'], section.id)

    // Membership lives on the room, not on any bot's profile: no bot-meta
    // write happens for a group filing.
    expect($groupChats.get()['Team Chat']?.sectionId).toBe(section.id)
    expect(saveBotMeta).not.toHaveBeenCalled()

    // The durable record in plugin storage carries it — the same record a
    // reload hydrates rooms from, so the filing survives a restart.
    const durable = storage.get('group-chats') as Record<string, { sectionId?: null | string }>
    expect(durable['Team Chat']?.sectionId).toBe(section.id)

    // Unfiling clears the assignment on the room and persists the clear.
    moveGroupChatsToSection(['Team Chat'], null)
    expect(groupChatSectionId('Team Chat', $groupChats.get())).toBeNull()
    expect((storage.get('group-chats') as Record<string, { sectionId?: null | string }>)['Team Chat']?.sectionId).toBeNull()
  })

  it('grouping seats group rows per their room record section; unfiled or dangling groups fall to Unassigned', () => {
    const section = createBotSection('XYZ')!

    moveGroupChatsToSection(['Team Chat'], section.id)
    // A room whose section was deleted keeps a dangling id — it must land in
    // Unassigned, not vanish (same safety bots get).
    $groupChats.set({ ...$groupChats.get(), Ghosted: { sectionId: 'sec-deleted' } as never })
    $botMeta.set({ nanox: { sectionId: section.id } })

    const rows = [row('nanox'), groupRow('Team Chat'), groupRow('Ghosted'), groupRow('Room')]

    const blocks = groupRowsBySection(rows, $botSections.get(), $botMeta.get(), $groupChats.get())

    expect(blocks.map(block => [block.key, block.rows.length])).toEqual([
      [`section:${section.id}`, 2],
      [UNASSIGNED_SECTION_KEY, 2]
    ])
    // Every row lands exactly once, groups included.
    expect(blocks.flatMap(block => block.rows)).toHaveLength(rows.length)
  })
})

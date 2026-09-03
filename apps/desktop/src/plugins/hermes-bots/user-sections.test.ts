/**
 * User sections — the three invariants that make membership-on-the-bot safe:
 * filing persists through `saveBotMeta` (so it rides profile sync), every row
 * lands in exactly one block with the remainder as Unassigned, and deleting a
 * section returns its bots to Unassigned rather than losing them.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

const { saveBotMeta, storage } = vi.hoisted(() => ({
  saveBotMeta: vi.fn<(bot: { name: string }, patch: Record<string, unknown>) => Promise<unknown>>(),
  storage: new Map<string, unknown>()
}))

vi.mock('./data', async () => {
  const { atom } = await import('nanostores')
  const $botMeta = atom<Record<string, { sectionId?: null | string }>>({})

  saveBotMeta.mockImplementation(async (bot: { name: string }, patch: Record<string, unknown>) => {
    $botMeta.set({ ...$botMeta.get(), [bot.name]: { ...$botMeta.get()[bot.name], ...patch } })

    return { serverOutcome: 'persisted', serverPersisted: true }
  })

  return { $botMeta, saveBotMeta }
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

import { $botMeta } from './data'
import type { RosterRow } from './types'
import {
  $botSections,
  createBotSection,
  deleteBotSection,
  groupRowsBySection,
  loadBotSections,
  moveBotsToSection,
  UNASSIGNED_SECTION_KEY
} from './user-sections'

const bot = (name: string) => ({ name }) as RosterRow
const row = (name: string) => ({ bot: bot(name), kind: 'bot' as const })

beforeEach(() => {
  storage.clear()
  $botMeta.set({})
  $botSections.set([])
  saveBotMeta.mockClear()
})

describe('user sections', () => {
  it('filing writes one sectionId per bot through saveBotMeta and survives a reload', async () => {
    const section = createBotSection('Clients', [bot('nanox'), bot('scout')])!

    // Membership rides the bot's own meta write (profile ui_meta), one per bot.
    await vi.waitFor(() => expect(saveBotMeta).toHaveBeenCalledTimes(2))
    expect(saveBotMeta).toHaveBeenCalledWith(bot('nanox'), { sectionId: section.id })

    // A no-op move (already there) writes nothing.
    await moveBotsToSection([bot('nanox')], section.id)
    expect(saveBotMeta).toHaveBeenCalledTimes(2)

    // The section record itself persists in plugin storage.
    $botSections.set([])
    loadBotSections()
    expect($botSections.get()).toEqual([{ id: section.id, name: 'Clients' }])
  })

  it('groups every row exactly once; unknown or missing sections fall to Unassigned, drawn last', () => {
    const rows = [row('nanox'), row('scout'), row('ghost'), { kind: 'group' as const, name: 'Room' }]

    const meta = {
      nanox: { sectionId: 'sec-clients' },
      scout: { sectionId: 'sec-workforce' },
      ghost: { sectionId: 'sec-deleted' }
    }

    const blocks = groupRowsBySection(
      rows,
      [
        { id: 'sec-clients', name: 'Clients' },
        { id: 'sec-workforce', name: 'Workforce' }
      ],
      meta
    )

    expect(blocks.map(b => [b.key, b.rows.length])).toEqual([
      ['section:sec-clients', 1],
      ['section:sec-workforce', 1],
      [UNASSIGNED_SECTION_KEY, 2]
    ])
    expect(blocks.flatMap(b => b.rows)).toHaveLength(rows.length)
    expect(groupRowsBySection(rows, [], meta)).toEqual([{ id: null, key: UNASSIGNED_SECTION_KEY, name: '', rows }])
  })

  it('deleting a section returns its bots to Unassigned, and undo refiles them', async () => {
    const section = createBotSection('Clients', [bot('nanox')])!
    createBotSection('Team')
    await vi.waitFor(() => expect($botMeta.get().nanox?.sectionId).toBe(section.id))

    const { members, undo } = deleteBotSection(section.id, [bot('nanox'), bot('scout')])

    expect(members).toEqual([bot('nanox')])
    expect($botSections.get().map(s => s.name)).toEqual(['Team'])
    await vi.waitFor(() => expect($botMeta.get().nanox?.sectionId).toBeNull())

    undo()
    expect($botSections.get().map(s => s.name)).toEqual(['Clients', 'Team'])
    await vi.waitFor(() => expect($botMeta.get().nanox?.sectionId).toBe(section.id))
  })
})

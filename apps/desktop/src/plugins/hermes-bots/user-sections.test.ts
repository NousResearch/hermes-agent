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
  const $botMeta = atom<Record<string, { sectionId?: null | string; sectionName?: null | string }>>({})

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
  adoptBotSectionsFromMeta,
  backfillBotSectionNames,
  createBotSection,
  deleteBotSection,
  dropBotSection,
  groupRowsBySection,
  loadBotSections,
  moveBotsToSection,
  renameBotSection,
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
    expect(saveBotMeta).toHaveBeenCalledWith(bot('nanox'), { sectionId: section.id, sectionName: 'Clients' })

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

  it('a second desktop rebuilds sections it never created from the id + name on each member', () => {
    // This machine has no section records, only the members' synced ui_meta.
    const meta = {
      nanox: { sectionId: 'sec-clients', sectionName: 'Clients' },
      scout: { sectionId: 'sec-clients', sectionName: 'Clients' },
      ghost: { sectionId: 'sec-legacy' } // filed before names rode along: nothing to draw
    }

    adoptBotSectionsFromMeta([bot('nanox'), bot('scout'), bot('ghost')], meta)
    expect($botSections.get()).toEqual([{ id: 'sec-clients', name: 'Clients' }])
    expect(storage.get('bot-sections-v1')).toEqual([{ id: 'sec-clients', name: 'Clients' }])

    // A known section takes the members' name only once every member agrees
    // on it — a rename elsewhere, fully stamped. A half-stamped rename leaves
    // the local name alone.
    adoptBotSectionsFromMeta([bot('nanox'), bot('scout')], {
      nanox: { sectionId: 'sec-clients', sectionName: 'Customers' },
      scout: { sectionId: 'sec-clients', sectionName: 'Clients' }
    })
    expect($botSections.get()).toEqual([{ id: 'sec-clients', name: 'Clients' }])
    adoptBotSectionsFromMeta([bot('nanox'), bot('scout')], {
      nanox: { sectionId: 'sec-clients', sectionName: 'Customers' },
      scout: { sectionId: 'sec-clients', sectionName: 'Customers' }
    })
    expect($botSections.get()).toEqual([{ id: 'sec-clients', name: 'Customers' }])
  })

  it('the desktop that knows a section backfills the name onto members filed before names rode along', async () => {
    // This machine made "Clients" before sectionName existed: the record is
    // local, the members carry only the id. Another desktop cannot rebuild
    // the section from that — so stamp the name here, once per member.
    $botSections.set([{ id: 'sec-clients', name: 'Clients' }])
    const meta = {
      nanox: { sectionId: 'sec-clients' },
      scout: { sectionId: 'sec-clients', sectionName: 'Clients' }, // already stamped
      ghost: { sectionId: 'sec-unknown' } // nobody here knows that section: nothing to stamp
    }
    $botMeta.set(meta)

    expect(backfillBotSectionNames([bot('nanox'), bot('scout'), bot('ghost')], meta).map(b => b.name)).toEqual(['nanox'])
    await vi.waitFor(() => expect(saveBotMeta).toHaveBeenCalledTimes(1))
    expect(saveBotMeta).toHaveBeenCalledWith(bot('nanox'), { sectionId: 'sec-clients', sectionName: 'Clients' })

    // The write set the name, so the next roster pass has nothing left to do.
    saveBotMeta.mockClear()
    expect(backfillBotSectionNames([bot('nanox'), bot('scout'), bot('ghost')], $botMeta.get())).toEqual([])
    expect(saveBotMeta).not.toHaveBeenCalled()
  })

  it('renaming re-stamps the members so the new name reaches other desktops', async () => {
    const section = createBotSection('Clients', [bot('nanox')])!
    await vi.waitFor(() => expect($botMeta.get().nanox?.sectionName).toBe('Clients'))
    saveBotMeta.mockClear()

    renameBotSection(section.id, 'Customers', [bot('nanox'), bot('scout')])
    await vi.waitFor(() => expect(saveBotMeta).toHaveBeenCalledTimes(1))
    expect(saveBotMeta).toHaveBeenCalledWith(bot('nanox'), { sectionId: section.id, sectionName: 'Customers' })
    expect($botSections.get()).toEqual([{ id: section.id, name: 'Customers' }])
  })

  it('dropping a section heading before/after another reorders and persists', () => {
    const a = createBotSection('Alpha')!
    const b = createBotSection('Beta')!
    const c = createBotSection('Gamma')!
    expect($botSections.get().map(s => s.name)).toEqual(['Alpha', 'Beta', 'Gamma'])

    // Drop Gamma BEFORE Beta: it lands between Alpha and Beta.
    dropBotSection(c.id, b.id, false)
    expect($botSections.get().map(s => s.name)).toEqual(['Alpha', 'Gamma', 'Beta'])

    // Drop Alpha AFTER Beta: it moves to the end.
    dropBotSection(a.id, b.id, true)
    expect($botSections.get().map(s => s.name)).toEqual(['Gamma', 'Beta', 'Alpha'])

    // A drop on itself and a drop naming an unknown id are no-ops.
    dropBotSection(a.id, a.id, true)
    dropBotSection(a.id, 'sec-missing', true)
    expect($botSections.get().map(s => s.name)).toEqual(['Gamma', 'Beta', 'Alpha'])

    // The persisted store carries the new order (what the next window loads).
    expect(storage.get('bot-sections-v1')).toEqual($botSections.get())
  })
})

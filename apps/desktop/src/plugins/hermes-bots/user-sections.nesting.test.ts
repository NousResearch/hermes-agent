/**
 * Nested sections — folders inside folders (Studio › Team, Studio › Clients).
 * The hierarchy is a `parentId` on the local section record; the invariants:
 * persisted input can never produce a loop or hide a folder, a move that would
 * make a loop is refused, deleting a folder moves everything under it up one
 * level (and Undo puts it all back), and records written before nesting
 * existed keep rendering exactly as before.
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
  const $botMeta = atom<Record<string, { sectionId?: null | string; sectionName?: null | string }>>({})

  saveBotMeta.mockImplementation(async (bot: { name: string }, patch: Record<string, unknown>) => {
    $botMeta.set({ ...$botMeta.get(), [bot.name]: { ...$botMeta.get()[bot.name], ...patch } })

    return { serverOutcome: 'persisted', serverPersisted: true }
  })

  return {
    $botMeta,
    $lastRoster: atom<unknown[]>([]),
    botRosterKey: (bot: { name?: string }) => `legacy::${bot?.name}`,
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

import { $botMeta } from './data'
import { $groupChats } from './group-chat'
import type { RosterRow } from './types'
import {
  $botSections,
  adoptBotSectionsFromMeta,
  BOT_SECTIONS_KEY,
  botSectionPath,
  createBotSection,
  deleteBotSection,
  groupChatSectionId,
  groupRowsBySection,
  loadBotSections,
  moveBotSection,
  moveBotSectionToParent,
  moveBotsToSection,
  moveGroupChatsToSection,
  normalizeBotSections,
  placeBotSectionBefore,
  resetBotSectionDeletes,
  SECTION_DRAG_PREFIX,
  UNASSIGNED_SECTION_KEY,
  visibleSectionBlocks
} from './user-sections'

const bot = (name: string) => ({ name }) as RosterRow
const row = (name: string) => ({ bot: bot(name), kind: 'bot' as const })

/** The next saveBotMeta applies locally at once (as the real one does) but
 *  its server half stays in flight until the returned release is called. */
function holdNextSave(): () => void {
  let release = () => {}
  const gate = new Promise<void>(resolve => (release = resolve))
  const apply = saveBotMeta.getMockImplementation()!

  saveBotMeta.mockImplementationOnce(async (owner, patch) => {
    const result = await apply(owner, patch)
    await gate

    return result
  })

  return () => release()
}

/** Studio › { Team › Engineering, Clients }, plus a top-level Personal. */
function studioTree() {
  const studio = createBotSection('Studio')!
  const team = createBotSection('Team', [], studio.id)!
  const engineering = createBotSection('Engineering', [], team.id)!
  const clients = createBotSection('Clients', [], studio.id)!
  const personal = createBotSection('Personal')!

  return { studio, clients, engineering, personal, team }
}

beforeEach(() => {
  storage.clear()
  $botMeta.set({})
  $botSections.set([])
  $groupChats.set({})
  resetBotSectionDeletes()
  saveBotMeta.mockClear()
})

describe('nested sections — persisted input', () => {
  it('flat records from before nesting load unchanged, and a nested tree round-trips through storage', () => {
    storage.set(BOT_SECTIONS_KEY, [
      { id: 'sec-a', name: 'Clients' },
      { id: 'sec-b', name: 'Team' }
    ])
    loadBotSections()
    expect($botSections.get()).toEqual([
      { id: 'sec-a', name: 'Clients' },
      { id: 'sec-b', name: 'Team' }
    ])

    const { studio, team } = studioTree()
    const saved = $botSections.get()

    $botSections.set([])
    loadBotSections()
    expect($botSections.get()).toEqual(saved)
    expect($botSections.get().find(s => s.id === team.id)?.parentId).toBe(studio.id)
  })

  it.each([
    ['a parent that does not exist', [{ id: 'a', name: 'A', parentId: 'gone' }]],
    ['a section that is its own parent', [{ id: 'a', name: 'A', parentId: 'a' }]],
    ['a parent id that is not a string', [{ id: 'a', name: 'A', parentId: 42 }]],
    ['a blank parent id', [{ id: 'a', name: 'A', parentId: '  ' }]]
  ])('%s lifts the section to the top level', (_label, input) => {
    expect(normalizeBotSections(input)).toEqual([{ id: 'a', name: 'A' }])
  })

  it('a loop in persisted parents is broken without losing a section, and every section still renders', () => {
    const sections = normalizeBotSections([
      { id: 'a', name: 'A', parentId: 'c' },
      { id: 'b', name: 'B', parentId: 'a' },
      { id: 'c', name: 'C', parentId: 'b' }
    ])

    expect(sections.map(s => s.id)).toEqual(['a', 'b', 'c'])
    expect(sections.filter(s => !s.parentId)).toHaveLength(1)
    expect(groupRowsBySection([], sections, {}).map(block => block.id)).toHaveLength(4)
  })
})

describe('nested sections — grouping', () => {
  it('blocks come depth-first with depth and a subtree total; every row lands exactly once', () => {
    const { studio, clients, engineering, personal, team } = studioTree()

    $botMeta.set({
      ceo: { sectionId: studio.id },
      dev1: { sectionId: engineering.id },
      dev2: { sectionId: engineering.id },
      pm: { sectionId: team.id },
      harbor: { sectionId: clients.id }
    })
    moveGroupChatsToSection(['Business'], team.id)

    const rows = [
      row('ceo'),
      row('dev1'),
      row('dev2'),
      row('pm'),
      row('harbor'),
      row('loose'),
      { kind: 'group' as const, name: 'Business' }
    ]

    const blocks = groupRowsBySection(rows, $botSections.get(), $botMeta.get(), $groupChats.get())

    expect(blocks.map(b => [b.name || b.key, b.depth, b.rows.length, b.total])).toEqual([
      ['Studio', 0, 1, 6],
      ['Team', 1, 2, 4],
      ['Engineering', 2, 2, 2],
      ['Clients', 1, 1, 1],
      ['Personal', 0, 0, 0],
      [UNASSIGNED_SECTION_KEY, 0, 1, 1]
    ])
    expect(blocks.flatMap(b => b.rows)).toHaveLength(rows.length)
    expect(blocks.find(b => b.id === engineering.id)?.parentId).toBe(team.id)
    expect(personal.parentId).toBeUndefined()
    expect(botSectionPath(engineering.id, $botSections.get())).toBe('Studio › Team › Engineering')
  })

  it('a subsection asked for under an unknown parent is created at the top level', () => {
    expect(createBotSection('Orphan', [], 'sec-missing')).toEqual({ id: expect.any(String), name: 'Orphan' })
  })
})

describe('nested sections — moves', () => {
  it('refuses to nest a section under itself, a descendant, or an unknown folder', () => {
    const { studio, engineering, team } = studioTree()
    const before = $botSections.get()

    expect(moveBotSectionToParent(studio.id, studio.id)).toBe(false)
    expect(moveBotSectionToParent(studio.id, team.id)).toBe(false)
    expect(moveBotSectionToParent(studio.id, engineering.id)).toBe(false)
    expect(moveBotSectionToParent(team.id, 'sec-missing')).toBe(false)
    expect($botSections.get()).toBe(before)
  })

  it('moving a folder carries its subtree and members, and persists; top level un-nests it', async () => {
    const { engineering, personal, team } = studioTree()

    await moveBotsToSection([bot('dev1')], engineering.id)
    expect(moveBotSectionToParent(team.id, personal.id)).toBe(true)
    expect(botSectionPath(engineering.id, $botSections.get())).toBe('Personal › Team › Engineering')
    expect($botMeta.get().dev1?.sectionId).toBe(engineering.id)
    expect(normalizeBotSections(storage.get(BOT_SECTIONS_KEY))).toEqual($botSections.get())

    expect(moveBotSectionToParent(team.id, null)).toBe(true)
    expect($botSections.get().find(s => s.id === team.id)?.parentId).toBeUndefined()
  })

  it('move up/down reorders among siblings only', () => {
    const { studio, clients, personal, team } = studioTree()

    moveBotSection(clients.id, -1)

    const order = (parentId?: string) =>
      $botSections
        .get()
        .filter(s => s.parentId === parentId)
        .map(s => s.name)

    expect(order(studio.id)).toEqual(['Clients', 'Team'])
    expect(order()).toEqual(['Studio', 'Personal'])

    // First among its siblings: nothing moves, even though another section
    // sits before it in the flat list.
    moveBotSection(clients.id, -1)
    expect(order(studio.id)).toEqual(['Clients', 'Team'])
    moveBotSection(personal.id, -1)
    expect(order()).toEqual(['Personal', 'Studio'])
    expect(team.parentId).toBe(studio.id)
  })
})

describe('nested sections — delete', () => {
  it('deleting a folder moves its bots, group chats and subsections up one level; undo restores all of it', async () => {
    const { studio, engineering, team } = studioTree()

    await moveBotsToSection([bot('pm')], team.id)
    moveGroupChatsToSection(['Business'], team.id)
    const roster = [bot('pm'), bot('other')]

    const { members, moved, parent, undo } = deleteBotSection(team.id, roster)

    expect(members).toEqual([bot('pm')])
    // pm, the Business room and the Engineering subsection: what the toast counts.
    expect(moved).toBe(3)
    expect(parent?.id).toBe(studio.id)
    expect($botSections.get().find(s => s.id === engineering.id)?.parentId).toBe(studio.id)
    expect(groupChatSectionId('Business', $groupChats.get())).toBe(studio.id)
    await vi.waitFor(() => expect($botMeta.get().pm).toEqual({ sectionId: studio.id, sectionName: 'Studio' }))

    undo()
    expect(botSectionPath(engineering.id, $botSections.get())).toBe('Studio › Team › Engineering')
    expect(groupChatSectionId('Business', $groupChats.get())).toBe(team.id)
    await vi.waitFor(() => expect($botMeta.get().pm).toEqual({ sectionId: team.id, sectionName: 'Team' }))
  })

  it('deleting a top-level folder lifts its subsections to the top level and clears its bots to Unassigned', async () => {
    const { studio, clients, team } = studioTree()

    await moveBotsToSection([bot('ceo')], studio.id)
    const { parent } = deleteBotSection(studio.id, [bot('ceo')])

    expect(parent).toBeNull()
    expect(
      $botSections
        .get()
        .filter(s => !s.parentId)
        .map(s => s.id)
    ).toEqual([team.id, clients.id, expect.any(String)])
    await vi.waitFor(() => expect($botMeta.get().ceo?.sectionId).toBeNull())
  })

  it('undo during a slow delete keeps every member in the restored folder', async () => {
    const { studio, team } = studioTree()

    await moveBotsToSection([bot('pm'), bot('dev2')], team.id)
    const release = holdNextSave()

    // pm's move up to Studio is stuck on a slow profile write; Undo lands
    // before the delete has walked to dev2.
    deleteBotSection(team.id, [bot('pm'), bot('dev2')]).undo()
    release()

    await vi.waitFor(() => expect($botMeta.get().pm?.sectionId).toBe(team.id))
    await new Promise(resolve => setTimeout(resolve, 0))
    expect($botMeta.get().dev2?.sectionId).toBe(team.id)
    expect(saveBotMeta).not.toHaveBeenCalledWith(bot('dev2'), expect.objectContaining({ sectionId: studio.id }))
  })

  it('deleting the parent while a child delete is still clearing never brings the child back', async () => {
    const { studio, team } = studioTree()

    await moveBotsToSection([bot('pm'), bot('dev2')], team.id)
    const roster = [bot('pm'), bot('dev2')]
    const release = holdNextSave()

    // pm's move up into Studio is stuck; Studio is deleted before the clear
    // reaches dev2, so the clear has to re-aim at the top level.
    deleteBotSection(team.id, roster)
    deleteBotSection(studio.id, roster)
    release()

    await vi.waitFor(() => expect($botMeta.get().dev2?.sectionId).toBeNull())
    await vi.waitFor(() => expect($botMeta.get().pm?.sectionId).toBeNull())
    adoptBotSectionsFromMeta(roster, $botMeta.get())
    expect($botSections.get().map(s => s.name)).not.toContain('Team')
    expect($botSections.get().map(s => s.name)).not.toContain('Studio')
  })

  it('stragglers of a child delete follow its parent to wherever the parent was moved before its own delete', async () => {
    const { engineering, personal, team } = studioTree()

    await moveBotsToSection([bot('dev1'), bot('dev2')], engineering.id)
    const roster = [bot('dev1'), bot('dev2')]
    const release = holdNextSave()

    // dev1's move up into Team is stuck. Meanwhile Team moves under Personal
    // and is deleted: its contents go to Personal, so dev2 must too.
    deleteBotSection(engineering.id, roster)
    moveBotSectionToParent(team.id, personal.id)
    deleteBotSection(team.id, roster)
    release()

    await vi.waitFor(() => expect($botMeta.get().dev2?.sectionId).toBe(personal.id))
    await vi.waitFor(() => expect($botMeta.get().dev1?.sectionId).toBe(personal.id))
  })

  it('deleting a section that is already gone changes nothing and does not redirect the first clear', async () => {
    const { studio, team } = studioTree()

    await moveBotsToSection([bot('pm'), bot('dev2')], team.id)
    const roster = [bot('pm'), bot('dev2')]
    const release = holdNextSave()

    deleteBotSection(team.id, roster)
    expect(deleteBotSection(team.id, roster)).toMatchObject({ members: [], moved: 0, parent: null })
    release()

    await vi.waitFor(() => expect($botMeta.get().dev2?.sectionId).toBe(studio.id))
    expect($botMeta.get().pm?.sectionId).toBe(studio.id)
  })

  it('undo leaves a subsection the user moved during the toast where they put it', () => {
    const { studio, engineering, team } = studioTree()

    const { undo } = deleteBotSection(team.id)

    moveBotSectionToParent(engineering.id, null)
    moveBotSectionToParent(studio.id, engineering.id)
    undo()

    expect(botSectionPath(team.id, $botSections.get())).toBe('Engineering › Studio › Team')
    expect($botSections.get().find(s => s.id === engineering.id)?.parentId).toBeUndefined()
  })
})

describe('nested sections — drag-and-drop sorting', () => {
  // The sidebar as drawn: depth-first, siblings in order.
  const layout = () =>
    groupRowsBySection([], $botSections.get(), {})
      .filter(block => block.id)
      .map(block => botSectionPath(block.id!, $botSections.get()))

  it("a folder dropped on another lands just before it, at that folder's level, carrying its subtree", () => {
    const { clients, personal, team } = studioTree()

    expect(placeBotSectionBefore(clients.id, team.id)).toBe(true)
    expect(placeBotSectionBefore(team.id, personal.id)).toBe(true)
    expect(layout()).toEqual(['Studio', 'Studio › Clients', 'Team', 'Team › Engineering', 'Personal'])
    expect(normalizeBotSections(storage.get(BOT_SECTIONS_KEY))).toEqual($botSections.get())
  })

  it('refuses a drop onto the folder itself, anything inside it, or an unknown folder', () => {
    const { studio, engineering, team } = studioTree()
    const before = $botSections.get()

    expect(placeBotSectionBefore(studio.id, studio.id)).toBe(false)
    expect(placeBotSectionBefore(studio.id, team.id)).toBe(false)
    expect(placeBotSectionBefore(studio.id, engineering.id)).toBe(false)
    expect(placeBotSectionBefore(team.id, 'sec-missing')).toBe(false)
    expect($botSections.get()).toBe(before)
  })

  it('a folder dropped on Unassigned goes to the end of the top level', () => {
    const { studio } = studioTree()

    moveBotSectionToParent(studio.id, null)
    expect(
      $botSections
        .get()
        .filter(s => !s.parentId)
        .map(s => s.name)
    ).toEqual(['Personal', 'Studio'])
  })
})

describe('nested sections — what the roster draws', () => {
  const blocksFor = () => groupRowsBySection([row('pm')], $botSections.get(), $botMeta.get())
  const names = (blocks: { id: null | string; name: string }[]) => blocks.map(b => b.name || 'Unassigned')
  const open = () => false

  it('a collapsed folder hides its whole subtree but stays visible itself', () => {
    const { team } = studioTree()

    expect(
      names(visibleSectionBlocks(blocksFor(), { collapsed: id => id === team.id, dragging: null, nested: false }))
    ).toEqual(['Studio', 'Team', 'Clients', 'Personal', 'Unassigned'])
  })

  it('inside a gateway bucket empty folders only appear while something is dragged', async () => {
    const { engineering } = studioTree()

    await moveBotsToSection([bot('pm')], engineering.id)
    expect(names(visibleSectionBlocks(blocksFor(), { collapsed: open, dragging: null, nested: true }))).toEqual([
      'Studio',
      'Team',
      'Engineering'
    ])
    expect(visibleSectionBlocks(blocksFor(), { collapsed: open, dragging: 'legacy::x', nested: true })).toHaveLength(5)
  })

  it('an empty Unassigned is drawn only while a folder is dragged, as the end-of-top-level target', async () => {
    const { personal } = studioTree()

    await moveBotsToSection([bot('pm')], personal.id)

    const visible = (dragging: null | string) =>
      names(visibleSectionBlocks(blocksFor(), { collapsed: open, dragging, nested: false }))

    expect(visible(null)).not.toContain('Unassigned')
    expect(visible('legacy::pm')).not.toContain('Unassigned')
    expect(visible(`${SECTION_DRAG_PREFIX}${personal.id}`)).toContain('Unassigned')
  })
})

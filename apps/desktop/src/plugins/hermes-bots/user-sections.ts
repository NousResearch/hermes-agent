/**
 * USER SECTIONS — folders the user makes, not folders the topology makes.
 *
 * The roster already had sections (`roster-sections.tsx`), but only AUTOMATIC
 * ones: one per gateway connection, plus the group-chat bucket. Those answer
 * "where does this bot run", which is not the question you are asking when you
 * want two client bots filed together under "Clients".
 *
 * So this is a SECOND axis, and it composes with the first rather than
 * replacing it. Two deliberate choices:
 *
 *   * The membership lives on the BOT (`sectionId` in its ui_meta), not as a
 *     member list on the section. A bot can only be in one place, deleting a
 *     section cannot orphan anybody, and the assignment rides the same
 *     profile.yaml sync every other bot setting already uses — so sections
 *     follow the profile to another machine.
 *   * "Unassigned" is not a section. It is whatever is left, always drawn
 *     last, and it is where members of a deleted section land. With no
 *     sections at all the roster renders exactly as it did before.
 *
 * Pure model + session atoms. No JSX — the pane composes it.
 */

import { atom } from 'nanostores'

import { $botMeta, saveBotMeta } from './data'
import { $groupChats, updateGroupChat } from './group-chat'
import { botRosterMeta } from './routing'
import { getPluginCtx } from './shared'
import type { BotMeta, GroupChat, RosterRow } from './types'

/** The one section-name dialog's state: New section (optionally filing the
 *  bot or group whose menu opened it) or Rename. */
export type SectionDialogState =
  | null
  | { bot?: RosterRow; group?: string; mode: 'create'; parentId?: string }
  | { id: string; mode: 'rename'; name: string }

export const UNASSIGNED_SECTION_KEY = 'section:unassigned'
export const BOT_SECTIONS_KEY = 'bot-sections-v1'

/**
 * A folder, optionally nested under another (`parentId`). The hierarchy is
 * this desktop's own layout, like order and empty sections: members carry
 * only their own folder's id + name, so a desktop that rebuilds a section
 * from member meta files it at the top level. Records written before nesting
 * existed have no `parentId` and stay top-level.
 */
export interface BotSection {
  id: string
  name: string
  parentId?: string
}

/** `[{ id, name, parentId? }]`, siblings in display order. */
export const $botSections = atom<BotSection[]>([])

/** Sections whose delete is still clearing its members. A delete clears them
 *  one profile write at a time, and a slow write (a remote gateway, a pooled
 *  backend still starting) leaves the rest carrying the id AND the name for
 *  seconds — long enough for adoptBotSectionsFromMeta to rebuild the section
 *  the user just deleted, under whatever name a not-yet-cleared member still
 *  carries. The entry expires when that clear finishes (after it, a member
 *  carrying the id was filed there again — by another desktop — and adopting
 *  it is right); Undo takes it out early. The token keeps an earlier delete's
 *  clear from expiring a later delete of the same id. */
const pendingSectionDeletes = new Map<string, symbol>()

/** Where each deleted section sent its contents (its parent at delete time,
 *  `null` for top level). A child's delete still clearing members when its
 *  parent is deleted follows this chain, so stragglers land where the
 *  parent's own contents went rather than somewhere stale. Undo removes the
 *  entry. */
const deletedSectionParents = new Map<string, null | string>()

/** Test seam. */
export function resetBotSectionDeletes(): void {
  pendingSectionDeletes.clear()
  deletedSectionParents.clear()
}

/** Roster key of the bot in flight during a drag. Session-only, and cleared
 *  on dragend even when the drop lands outside any target — a stuck
 *  "dragging" state outlives the gesture and reads as a broken pane. */
export const $draggingBot = atom<null | string>(null)

export function normalizeBotSections(value: unknown): BotSection[] {
  if (!Array.isArray(value)) {
    return []
  }

  const seen = new Set<string>()
  const out: BotSection[] = []

  for (const entry of value) {
    const id = String((entry as BotSection)?.id || '').trim()
    const name = String((entry as BotSection)?.name || '').trim()
    const rawParent = (entry as BotSection)?.parentId
    const parentId = typeof rawParent === 'string' ? rawParent.trim() : ''

    if (!id || !name || seen.has(id)) {
      continue
    }

    seen.add(id)
    out.push(parentId ? { id, name, parentId } : { id, name })
  }

  // Persisted input is untrusted: a parent that does not exist, or a chain
  // that loops back on itself, would hide a folder forever. Either lifts the
  // folder to the top level — the first entry found closing a loop is lifted,
  // which breaks the loop for every other member of it.
  const byId = new Map(out.map(section => [section.id, section]))

  for (const section of out) {
    if (section.parentId && (!byId.has(section.parentId) || isSectionWithin(section.parentId, section.id, byId))) {
      delete section.parentId
    }
  }

  return out
}

/** Whether `id` is `ancestorId` or sits anywhere below it. Bounded by the map
 *  size, so a loop in unvalidated input cannot spin. */
function isSectionWithin(id: string, ancestorId: string, byId: Map<string, BotSection>): boolean {
  let cursor: string | undefined = id

  for (let hops = 0; cursor && hops <= byId.size; hops++) {
    if (cursor === ancestorId) {
      return true
    }

    cursor = byId.get(cursor)?.parentId
  }

  return false
}

/** "Studio › Team" — a folder's name with its ancestors, for menus that list
 *  every folder flat. */
export function botSectionPath(id: string, sections: BotSection[]): string {
  const byId = new Map(sections.map(section => [section.id, section]))
  const names: string[] = []
  let cursor = byId.get(id)

  while (cursor && names.length <= byId.size) {
    names.unshift(cursor.name)
    cursor = cursor.parentId ? byId.get(cursor.parentId) : undefined
  }

  return names.join(' › ')
}

/** Folders `id` may move under: any folder except itself and its own
 *  descendants (that would make a loop). */
export function botSectionParentOptions(id: string, sections: BotSection[]): BotSection[] {
  const byId = new Map(sections.map(section => [section.id, section]))

  return sections.filter(section => !isSectionWithin(section.id, id, byId))
}

function persistBotSections(next: BotSection[]): void {
  $botSections.set(next)

  try {
    getPluginCtx()?.storage?.set?.(BOT_SECTIONS_KEY, next)
  } catch {
    // No storage — sections live for this window only, which is strictly
    // better than the pane throwing while the user drags a bot into a folder.
  }
}

/** Read the persisted list back at plugin start. */
export function loadBotSections(): void {
  try {
    $botSections.set(normalizeBotSections(getPluginCtx()?.storage?.get?.(BOT_SECTIONS_KEY, [])))
  } catch {
    $botSections.set([])
  }
}

function newSectionId(): string {
  return `sec-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`
}

/** Create a section (under `parentId` when it names a known folder) and file
 *  `bots` into it. Returns the new section, or null when the name is blank. */
export function createBotSection(name: string, bots: RosterRow[] = [], parentId?: string): BotSection | null {
  const clean = String(name || '').trim()

  if (!clean) {
    return null
  }

  const parent = parentId && $botSections.get().some(s => s.id === parentId) ? parentId : null

  const section: BotSection = parent
    ? { id: newSectionId(), name: clean, parentId: parent }
    : { id: newSectionId(), name: clean }

  persistBotSections([...$botSections.get(), section])
  void moveBotsToSection(bots, section.id)

  return section
}

export function renameBotSection(id: string, name: string, roster: RosterRow[] = []): void {
  const clean = String(name || '').trim()

  if (!clean) {
    return
  }

  persistBotSections($botSections.get().map(s => (s.id === id ? { ...s, name: clean } : s)))
  // Members carry the name with their membership (see moveBotsToSection), so
  // a rename re-stamps them — that is how the new name reaches other desktops.
  void moveBotsToSection(
    (roster || []).filter(bot => botSectionId(bot, $botMeta.get()) === id),
    id
  )
}

/**
 * Sections another desktop made. The section RECORD lives in that machine's
 * plugin storage, but every member's ui_meta reaches this machine carrying the
 * section's id AND name — so rebuild the records we have never seen, and the
 * roster draws the same folders here. A known section takes the members' name
 * only when every member agrees on it (a rename elsewhere, fully stamped);
 * while members disagree — a re-stamp still in flight — the local name stays.
 * Order and empty sections remain this desktop's own. The roster pane calls
 * this on every roster/meta change.
 */
export function adoptBotSectionsFromMeta(roster: RosterRow[], metaByName: Record<string, BotMeta>): void {
  const names = new Map<string, Set<string>>()

  for (const bot of roster || []) {
    const meta = bot ? botRosterMeta(bot, metaByName) : null
    const id = meta?.sectionId ? String(meta.sectionId) : null
    const name = String(meta?.sectionName || '').trim()

    if (id && name) {
      names.set(id, (names.get(id) || new Set()).add(name))
    }
  }

  const agreed = (id: string): null | string => {
    const set = names.get(id)

    return set?.size === 1 ? [...set][0]! : null
  }

  const local = $botSections.get()
  const known = new Set(local.map(s => s.id))
  const renamed = local.map(s => ({ ...s, name: agreed(s.id) ?? s.name }))

  const adopted = [...names.keys()]
    .filter(id => !known.has(id) && !pendingSectionDeletes.has(id))
    .map(id => ({ id, name: [...names.get(id)!][0]! }))

  const next = [...renamed, ...adopted]

  if (next.some((s, i) => s.id !== local[i]?.id || s.name !== local[i]?.name)) {
    persistBotSections(next)
  }
}

/**
 * The other half of adoptBotSectionsFromMeta: bots filed before the name rode
 * along carry only `sectionId`, and a desktop that never made that section
 * has nothing to rebuild it from. So the desktop that DOES know the section —
 * usually the one that created it — stamps the name onto every such member
 * through the same one-write-per-profile path filing uses. Returns the members
 * being stamped. Runs once per member: the write sets `sectionName`, so the
 * next roster pass finds nothing to do. Members of a section nobody here
 * knows are left alone — there is no name to give them.
 */
export function backfillBotSectionNames(roster: RosterRow[], metaByName: Record<string, BotMeta>): RosterRow[] {
  const known = new Set($botSections.get().map(s => s.id))
  const bySection = new Map<string, RosterRow[]>()

  for (const bot of roster || []) {
    const meta = bot ? botRosterMeta(bot, metaByName) : null
    const id = meta?.sectionId ? String(meta.sectionId) : null

    if (id && known.has(id) && !String(meta?.sectionName || '').trim()) {
      bySection.set(id, [...(bySection.get(id) || []), bot])
    }
  }

  for (const [id, members] of bySection) {
    void moveBotsToSection(members, id)
  }

  return [...bySection.values()].flat()
}

/** What a delete took apart, so Undo can put it back. */
interface DeletedSection {
  children: string[]
  groups: string[]
  index: number
  members: RosterRow[]
  parentId: null | string
  section: BotSection
}

/**
 * Delete the section only. Nothing under it is deleted or hidden: its bots,
 * group chats and subsections move up one level — into the parent folder, or
 * Unassigned / top level for a top-level section. That is the whole reason
 * membership lives on the item rather than on the section. Returns how many
 * items moved and an undo that puts the section back in its slot and refiles
 * the same bots, groups and subsections, so the delete needs no confirmation.
 */
export function deleteBotSection(
  id: string,
  roster: RosterRow[] = []
): { members: RosterRow[]; moved: number; parent: BotSection | null; undo: () => void } {
  const list = $botSections.get()
  const index = list.findIndex(s => s.id === id)
  const section = list[index]

  // Already gone (a second delete of the same id): nothing to take apart, and
  // re-arming the shield would send a clear still in flight the wrong way.
  if (!section) {
    return { members: [], moved: 0, parent: null, undo: () => {} }
  }

  const parentId = section.parentId ?? null
  const members = (roster || []).filter(bot => botSectionId(bot, $botMeta.get()) === id)
  const groups = Object.keys($groupChats.get()).filter(name => groupChatSectionId(name, $groupChats.get()) === id)
  const children = list.filter(s => s.parentId === id).map(s => s.id)
  const token = Symbol(id)

  pendingSectionDeletes.set(id, token)
  deletedSectionParents.set(id, parentId)
  persistBotSections(list.filter(s => s.id !== id).map(s => (s.parentId === id ? withParent(s, parentId) : s)))
  moveGroupChatsToSection(groups, parentId)
  void clearDeletedSection(id, token, members)

  return {
    members,
    moved: members.length + groups.length + children.length,
    parent: list.find(s => s.id === parentId) || null,
    undo: () => restoreDeletedSection({ children, groups, index, members, parentId, section })
  }
}

/** The live folder a deleted section's contents belong in now: its parent
 *  at delete time, following any parent deleted since to where that one sent
 *  its own contents; `null` = top level / Unassigned. */
function liveDestination(id: string): null | string {
  let target = deletedSectionParents.get(id) ?? null

  for (let hops = 0; target && hops <= deletedSectionParents.size; hops++) {
    if ($botSections.get().some(s => s.id === target)) {
      return target
    }

    target = deletedSectionParents.get(target) ?? null
  }

  return null
}

/**
 * Move a deleted section's bots into its live destination, and only then lift
 * the delete's shield against re-adoption. That destination can itself be
 * deleted mid-walk (the walk then stops early), so re-aim until no member
 * still carries the id — otherwise a member left behind would rebuild the
 * deleted section. Each re-aim follows one more deleted parent, so the passes
 * are bounded. Undo (or a later delete of the same id) retires the token and
 * ends the walk.
 */
async function clearDeletedSection(id: string, token: symbol, members: RosterRow[]) {
  const current = () => pendingSectionDeletes.get(id) === token

  try {
    for (let attempt = 0; attempt <= deletedSectionParents.size && current(); attempt++) {
      const stuck = members.filter(bot => botSectionId(bot, $botMeta.get()) === id)

      if (!stuck.length) {
        return
      }

      await moveBotsToSection(stuck, liveDestination(id), current)
    }
  } finally {
    if (current()) {
      pendingSectionDeletes.delete(id)
    }
  }
}

function restoreDeletedSection({ children, groups, index, members, parentId, section }: DeletedSection): void {
  pendingSectionDeletes.delete(section.id)
  deletedSectionParents.delete(section.id)

  const current = $botSections
    .get()
    .filter(s => s.id !== section.id)
    // Only subsections still where the delete put them: one the user moved
    // during the toast stays where they moved it.
    .map(s => (children.includes(s.id) && (s.parentId ?? null) === parentId ? withParent(s, section.id) : s))

  current.splice(Math.min(index, current.length), 0, section)
  persistBotSections(normalizeBotSections(current))
  moveGroupChatsToSection(groups, section.id)
  void moveBotsToSection(members, section.id)
}

function withParent(section: BotSection, parentId: null | string): BotSection {
  return parentId ? { ...section, parentId } : { id: section.id, name: section.name }
}

/** Reorder a section among its siblings (same parent). */
export function moveBotSection(id: string, delta: number): void {
  const list = $botSections.get()
  const parentId = list.find(s => s.id === id)?.parentId ?? null
  const siblings = list.filter(s => (s.parentId ?? null) === parentId).map(s => s.id)
  const from = siblings.indexOf(id)
  const to = from + delta

  if (from < 0 || to < 0 || to >= siblings.length) {
    return
  }

  // Swap the two siblings' slots in the flat list; everything else stays put.
  const a = list.findIndex(s => s.id === id)
  const b = list.findIndex(s => s.id === siblings[to])

  const next = list.slice()

  ;[next[a], next[b]] = [next[b]!, next[a]!]
  persistBotSections(next)
}

/**
 * Nest a section under another folder (`null` = top level). Refused, and
 * returns false, when the target is unknown, the section itself, or one of
 * its own descendants — any of those would make a loop. The section keeps its
 * members and subsections; it lands last among its new siblings.
 */
export function moveBotSectionToParent(id: string, parentId: null | string): boolean {
  const list = $botSections.get()
  const section = list.find(s => s.id === id)

  if (!section || (parentId && !botSectionParentOptions(id, list).some(s => s.id === parentId))) {
    return false
  }

  persistBotSections([...list.filter(s => s.id !== id), withParent(section, parentId)])

  return true
}

/**
 * Drag-and-drop sorting: put section `id` directly before `targetId`, as its
 * sibling (so a drop can also change level). Refused, returning false, when
 * either is unknown or the target sits inside the dragged section's subtree.
 */
export function placeBotSectionBefore(id: string, targetId: string): boolean {
  const list = $botSections.get()
  const section = list.find(s => s.id === id)
  const target = list.find(s => s.id === targetId)

  if (!section || !target || !botSectionParentOptions(id, list).some(s => s.id === targetId)) {
    return false
  }

  const rest = list.filter(s => s.id !== id)

  rest.splice(
    rest.findIndex(s => s.id === targetId),
    0,
    withParent(section, target.parentId ?? null)
  )
  persistBotSections(rest)

  return true
}

/** The drag payload prefix that marks an in-flight SECTION heading. */
export const SECTION_DRAG_PREFIX = 'section:'

/**
 * `null` clears the assignment (back to Unassigned). One `saveBotMeta` per
 * bot — membership is a field on each bot's own profile, so that IS one write
 * per profile — and the writes run in sequence rather than fanned out, so the
 * shared local snapshot is never committed by two saves at once. The section's
 * NAME rides along: the record itself is local to the desktop that made it,
 * and the name is what lets another desktop rebuild it (adoptBotSectionsFromMeta).
 * `stillWanted` lets a caller abandon the rest of the walk (a delete's clear
 * that Undo has overtaken).
 */
export async function moveBotsToSection(
  bots: RosterRow[],
  sectionId: null | string,
  stillWanted: () => boolean = () => true
): Promise<void> {
  const sectionName = (sectionId && $botSections.get().find(s => s.id === sectionId)?.name) || null

  for (const bot of bots || []) {
    if (!bot) {
      continue
    }

    // Deleted while this loop was still walking its members (a rename's
    // re-stamp stuck behind a slow write): filing the rest would put them back
    // into a section that no longer exists.
    if ((sectionId && !$botSections.get().some(s => s.id === sectionId)) || !stillWanted()) {
      return
    }

    const current = botRosterMeta(bot, $botMeta.get())

    if (botSectionId(bot, $botMeta.get()) !== (sectionId || null) || (current?.sectionName || null) !== sectionName) {
      await saveBotMeta(bot, { sectionId: sectionId || null, sectionName })
    }
  }
}

export function botSectionId(bot: RosterRow, metaByName: Record<string, BotMeta>): null | string {
  const id = botRosterMeta(bot, metaByName)?.sectionId

  return id ? String(id) : null
}

// ── group chats ──────────────────────────────────────────────────────────────
//
// A group chat files into a section by the same membership-on-the-item rule as
// a bot, but a group has no profile meta to carry the field — its room record
// is its only durable identity. So the assignment is a field on the room and
// rides the room's own persistence (plugin storage), which is also why it is
// deliberately NOT part of the bounded gateway sync projection: that mirror
// carries conversations to other clients, not this machine's sidebar layout —
// the same local scope the section list itself has.

/** The drag payload prefix that marks an in-flight GROUP row (vs a bot's
 *  roster key). */
export const GROUP_DRAG_PREFIX = 'group:'

export function groupDragKey(name: string): string {
  return `${GROUP_DRAG_PREFIX}${name}`
}

/** A group's section, off the live room record. */
export function groupChatSectionId(name: string, rooms: Record<string, GroupChat>): null | string {
  const id = rooms?.[name]?.sectionId

  return id ? String(id) : null
}

/** File `groups` into a section (`null` clears, back to the group-chat
 *  bucket). One room write per group whose assignment actually changes,
 *  through `updateGroupChat` so the durable record and the atom move together. */
export function moveGroupChatsToSection(groups: string[], sectionId: null | string): void {
  for (const name of groups || []) {
    if (!name || groupChatSectionId(name, $groupChats.get()) === (sectionId || null)) {
      continue
    }

    updateGroupChat(
      name,
      (room: GroupChat) => ({
        ...room,
        sectionId: sectionId || null
      }),
      // Layout-only: no reason to publish the conversation projection.
      { sync: false }
    )
  }
}

export interface SectionBlock<TRow> {
  /** Nesting level: 0 for a top-level section and for Unassigned. */
  depth: number
  id: null | string
  key: string
  name: string
  parentId: null | string
  rows: TRow[]
  /** Rows in this block and every block nested under it. */
  total: number
}

/**
 * Split roster rows into section blocks, depth-first (each section followed
 * by its subsections, siblings in section order), with Unassigned last. Pure, and returns EVERY row exactly once: a row whose `sectionId`
 * names a section that no longer exists lands in Unassigned rather than
 * vanishing, which is what makes deleting a section safe. Group rows seat per
 * their room record's `sectionId` (`rooms`); a group with no room yet is
 * unassigned.
 */
export function groupRowsBySection<TRow extends { bot?: RosterRow } | RosterRow>(
  rows: TRow[],
  sections: unknown,
  metaByName: Record<string, BotMeta>,
  groupRooms: Record<string, GroupChat> = {}
): SectionBlock<TRow>[] {
  const list = normalizeBotSections(sections)
  const known = new Set(list.map(s => s.id))
  const byId = new Map<string, TRow[]>(list.map(s => [s.id, [] as TRow[]]))
  const loose: TRow[] = []

  for (const row of rows || []) {
    const group = (row as { kind?: string; name?: string })?.kind === 'group' ? row : null

    const id = group
      ? groupChatSectionId(String((group as { name?: string }).name || ''), groupRooms)
      : botSectionId(((row as { bot?: RosterRow })?.bot || row) as RosterRow, metaByName)

    if (id && known.has(id)) {
      byId.get(id)!.push(row)
    } else {
      loose.push(row)
    }
  }

  const blocks: SectionBlock<TRow>[] = []

  const visit = (parentId: null | string, depth: number): number => {
    let sum = 0

    for (const section of list.filter(s => (s.parentId ?? null) === parentId)) {
      const rows = byId.get(section.id) || []

      const block = {
        depth,
        id: section.id,
        key: `section:${section.id}`,
        name: section.name,
        parentId,
        rows,
        total: 0
      }

      blocks.push(block)
      block.total = rows.length + visit(section.id, depth + 1)
      sum += block.total
    }

    return sum
  }

  visit(null, 0)
  blocks.push({
    depth: 0,
    id: null,
    key: UNASSIGNED_SECTION_KEY,
    name: '',
    parentId: null,
    rows: loose,
    total: loose.length
  })

  return blocks
}

export interface VisibleBlocksOptions {
  /** Whether the block with this section id is folded. */
  collapsed: (id: string) => boolean
  /** The roster drag key in flight, if any. */
  dragging: null | string
  /** Drawn inside a gateway bucket. */
  nested: boolean
}

/**
 * Which of `groupRowsBySection`'s blocks the roster draws. An empty
 * Unassigned is not worth a heading — except while a FOLDER is dragged, when
 * it is the "end of the top level" drop target. An empty NAMED section is,
 * because it is somewhere the user made and is about to drop into; inside a
 * gateway bucket the same empty section would repeat under every connection,
 * so there it only appears while a drag is in flight. A folder counts as
 * non-empty when anything below it is. Blocks arrive depth-first, so a
 * collapsed (or hidden) folder hides its subtree by hiding every block whose
 * parent is hidden or collapsed.
 */
export function visibleSectionBlocks<TRow>(
  blocks: SectionBlock<TRow>[],
  { collapsed, dragging, nested }: VisibleBlocksOptions
): SectionBlock<TRow>[] {
  const hidden = new Set<string>()
  const sorting = Boolean(dragging?.startsWith(SECTION_DRAG_PREFIX))

  return blocks.filter(block => {
    const wanted = block.id ? !nested || Boolean(dragging) : sorting
    const visible = !(block.parentId && hidden.has(block.parentId)) && Boolean(block.total || wanted)

    if (block.id && (!visible || collapsed(block.id))) {
      hidden.add(block.id)
    }

    return visible
  })
}

// ── drag and drop ────────────────────────────────────────────────────────────
//
// Filing a bot by dragging it onto a section, which is the gesture people
// reach for first; the row's "Move to section" submenu is the same action
// for anyone who does not.
//
// A CUSTOM MIME TYPE, not `text/plain`: the roster shares a window with the
// composer, the transcript and the tab strip, all of which accept dropped
// text. A private type means a bot dragged onto any of them is simply not a
// valid payload there, instead of pasting its roster key into someone's
// message. `dataTransfer.types` is readable during dragover (the DATA itself
// is not, by design), so a drop target can still light up correctly.

export const BOT_DRAG_MIME = 'application/x-hermes-bot-key'

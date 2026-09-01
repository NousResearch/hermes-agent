/**
 * USER SECTIONS — folders the user makes, not folders the topology makes.
 *
 * The roster already had sections (`roster-sections.tsx`), but only AUTOMATIC
 * ones: one per gateway connection, plus the group-chat bucket. Those answer
 * "where does this bot run", which is not the question you are asking when you
 * want NanoX and MODE filed together under "Clients".
 *
 * So this is a SECOND axis, and it composes with the first rather than
 * replacing it: the gateway sections still render exactly as they did whenever
 * the roster is showing more than one connection, and user sections group the
 * flat list underneath. Two deliberate choices, carried over from the branch
 * this is ported from:
 *
 *   * The membership lives on the BOT (`sectionId` in its ui_meta), not as a
 *     member list on the section. A bot can only be in one place, deleting a
 *     section cannot orphan anybody, and the assignment rides the same
 *     profile.yaml sync every other bot setting already uses — so sections
 *     follow the profile to another machine.
 *   * "Unassigned" is not a section. It is whatever is left, always drawn, and
 *     it is where members of a deleted section land. It has no record, so its
 *     collapsed state keys off this literal.
 *
 * Pure model + two session atoms. No JSX — the pane composes it.
 */

import { atom } from 'nanostores'

import { $botMeta, saveBotMeta } from './data'
import { botRosterMeta } from './routing'
import { getPluginCtx } from './shared'
import type { BotMeta, RosterRow } from './types'

export const UNASSIGNED_SECTION_KEY = 'section:unassigned'
export const BOT_SECTIONS_KEY = 'bot-sections-v1'

export interface BotSection {
  id: string
  name: string
  /** Draw the folder glyph beside the name. Default on; a user who wants a
   *  bare list of names can turn it off per section. Optional so every
   *  section persisted before this existed still reads as "show it". */
  icon?: boolean
}

/** `[{ id, name }]`, in display order. */
export const $botSections = atom<BotSection[]>([])

/** Roster keys the user has multi-selected (cmd/ctrl-click). Session-only: a
 *  selection is a gesture in progress, not a setting. */
export const $botPicked = atom<string[]>([])

/** The row a shift-click range extends FROM — the last plain click or the
 *  last end of a shift-range, mirroring how Finder/Mail anchor a range so a
 *  second shift-click re-anchors from where you are, not where you started. */
export const $botPickAnchor = atom<null | string>(null)

/**
 * The roster key of the bot being renamed in place, and the text in the field.
 *
 * MODULE state, not component state. It was `useState` inside `BotRow`, and
 * double-click did nothing: opening a bot resolves its source and canonical
 * chat, which changes `botRosterKey` — so the row REMOUNTS between the click
 * and the double-click, and the flag was gone before it could paint. The
 * handler fired every time; the state did not survive to the next render.
 * (Verified in the running app: the console log landed, `data-renaming` was
 * still "0".) Keying the caret outside the row is what makes it immune.
 */
export const $renamingBot = atom<null | string>(null)
export const $renamingBotDraft = atom('')

/** The section whose header is currently an editable name field. Session-only
 *  by nature: a rename in progress is a caret, not a setting. */
export const $renamingSection = atom<null | string>(null)

export function normalizeBotSections(value: unknown): BotSection[] {
  if (!Array.isArray(value)) {
    return []
  }

  const seen = new Set<string>()
  const out: BotSection[] = []

  for (const entry of value) {
    const id = String((entry as BotSection)?.id || '').trim()
    const name = String((entry as BotSection)?.name || '').trim()

    if (!id || seen.has(id)) {
      continue
    }

    seen.add(id)
    out.push({
      id,
      name: name || 'Section',
      // Only ever stored as an explicit false — absent means on.
      ...((entry as BotSection)?.icon === false ? { icon: false } : {})
    })
  }

  return out
}

export function persistBotSections(next: unknown): Promise<void> {
  const value = normalizeBotSections(next)

  $botSections.set(value)

  try {
    return Promise.resolve(getPluginCtx()?.storage?.set?.(BOT_SECTIONS_KEY, value))
      .then(() => undefined)
      .catch(() => undefined)
  } catch {
    // No storage — sections live for this window only, which is strictly
    // better than the pane throwing while the user drags a bot into a folder.
    return Promise.resolve()
  }
}

/** Read the persisted list back at plugin start. */
/**
 * The roster's three standing sections, with FIXED ids.
 *
 * Membership lives on each bot as `ui_meta.hermes-bots.sectionId`, which is a
 * file in the profile — but the section RECORDS live in plugin storage, which
 * is localStorage. Generated ids would mean the two halves could never be set
 * up together from outside the app: a profile.yaml written by hand would point
 * at a section id that does not exist, and the bot would silently land in
 * Unassigned. Fixed ids are what make the pairing writable from either side.
 *
 * Seeding is ADDITIVE and idempotent: a section already present by id is left
 * exactly as it is — including a rename, an icon setting, and its position —
 * and anything the user made themselves is untouched. Deleting one of these on
 * purpose is the one thing this cannot tell apart from never having had it, so
 * a deleted standing section comes back on next load; renaming it is the way
 * to make it yours.
 */
const SEEDED_SECTIONS: BotSection[] = [
  { id: 'sec-general', name: 'General' },
  { id: 'sec-workforce', name: 'Workforce' },
  { id: 'sec-clients', name: 'Clients' }
]

export async function loadBotSections(): Promise<void> {
  try {
    const stored = await Promise.resolve(getPluginCtx()?.storage?.get?.(BOT_SECTIONS_KEY, []))
    const list = normalizeBotSections(stored)
    const seeded = withSeededSections(list)

    $botSections.set(seeded)

    // Only write back when seeding actually added something, so an ordinary
    // load stays a read.
    if (seeded.length !== list.length) {
      void persistBotSections(seeded)
    }
  } catch {
    $botSections.set(normalizeBotSections(SEEDED_SECTIONS))
  }
}


function withSeededSections(list: BotSection[]): BotSection[] {
  const known = new Set(list.map(section => section.id))
  const missing = SEEDED_SECTIONS.filter(section => !known.has(section.id))

  // Seeded sections lead, in their declared order, so a fresh roster reads
  // General / Workforce / Clients rather than in load order.
  return missing.length ? [...missing, ...list] : list
}

function newSectionId(): string {
  return `sec-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`
}

/** Create a section and move `bots` into it. Returns the new section. */
export function createBotSection(name: string, bots: RosterRow[] = []): BotSection {
  const section: BotSection = { id: newSectionId(), name: String(name || '').trim() || 'New section' }

  void persistBotSections([...$botSections.get(), section])
  moveBotsToSection(bots, section.id)

  return section
}

/** Show or hide the folder glyph on one section's heading. */
export function setBotSectionIcon(id: string, icon: boolean): void {
  void persistBotSections($botSections.get().map(s => (s.id === id ? { ...s, icon } : s)))
}

export function renameBotSection(id: string, name: string): void {
  const clean = String(name || '').trim()

  if (!clean) {
    return
  }

  void persistBotSections($botSections.get().map(s => (s.id === id ? { ...s, name: clean } : s)))
}

/** Delete the section only. Its members are not deleted and not hidden — they
 *  fall back to Unassigned, which is the whole reason membership lives on the
 *  bot rather than on the section. */
export function deleteBotSection(id: string, roster: RosterRow[] = []): void {
  void persistBotSections($botSections.get().filter(s => s.id !== id))
  moveBotsToSection(
    (roster || []).filter(bot => botSectionId(bot, $botMeta.get()) === id),
    null
  )
}

export function moveBotSection(id: string, delta: number): void {
  const list = $botSections.get()
  const from = list.findIndex(s => s.id === id)
  const to = from + delta

  if (from < 0 || to < 0 || to >= list.length) {
    return
  }

  const next = list.slice()
  const [moved] = next.splice(from, 1)

  next.splice(to, 0, moved!)
  void persistBotSections(next)
}

/** `null` clears the assignment (back to Unassigned). */
export function moveBotsToSection(bots: RosterRow[], sectionId: null | string): void {
  for (const bot of bots || []) {
    if (bot) {
      void saveBotMeta(bot, { sectionId: sectionId || null })
    }
  }
}

export function botSectionId(bot: RosterRow, metaByName: Record<string, BotMeta>): null | string {
  const id = botRosterMeta(bot, metaByName)?.sectionId

  return id ? String(id) : null
}

export interface SectionBlock<TRow> {
  id: null | string
  key: string
  name: string
  rows: TRow[]
}

/**
 * Split roster rows into section blocks, in section order, with Unassigned
 * last. Pure, and returns EVERY row exactly once: a row whose `sectionId`
 * names a section that no longer exists lands in Unassigned rather than
 * vanishing, which is what makes deleting a section safe.
 */
export function groupRowsBySection<TRow extends { bot?: RosterRow } | RosterRow>(
  rows: TRow[],
  sections: unknown,
  metaByName: Record<string, BotMeta>
): SectionBlock<TRow>[] {
  const list = normalizeBotSections(sections)
  const known = new Set(list.map(s => s.id))
  const byId = new Map<string, TRow[]>(list.map(s => [s.id, [] as TRow[]]))
  const loose: TRow[] = []

  for (const row of rows || []) {
    const bot = ((row as { bot?: RosterRow })?.bot || row) as RosterRow
    const id = bot ? botSectionId(bot, metaByName) : null

    if (id && known.has(id)) {
      byId.get(id)!.push(row)
    } else {
      loose.push(row)
    }
  }

  const blocks: SectionBlock<TRow>[] = list.map(section => ({
    id: section.id,
    key: `section:${section.id}`,
    name: section.name,
    rows: byId.get(section.id) || []
  }))

  blocks.push({ id: null, key: UNASSIGNED_SECTION_KEY, name: 'Unassigned', rows: loose })

  return blocks
}

// ── drag and drop ────────────────────────────────────────────────────────────
//
// Filing a bot by dragging it onto a section heading, which is the gesture
// people reach for first and the one the context menu's "Move to section…" was
// standing in for.
//
// A CUSTOM MIME TYPE, not `text/plain`: the roster shares a window with the
// composer, the transcript and the tab strip, all of which accept dropped
// text. A private type means a bot dragged onto any of them is simply not a
// valid payload there, instead of pasting its roster key into someone's
// message. `dataTransfer.types` is readable during dragover (the DATA itself
// is not, by design), so a drop target can still light up correctly.

export const BOT_DRAG_MIME = 'application/x-hermes-bot-keys'

/** Roster keys in flight during a drag. Session-only, and cleared on dragend
 *  even when the drop lands outside any target — a stuck "dragging" highlight
 *  outlives the gesture and reads as a broken pane. */
export const $draggingBots = atom<string[]>([])

/** Keys being dragged, as a payload string. Multi-select drags the whole
 *  selection when the dragged row is part of it — same rule as the section
 *  context menu's `targets()`. */
export function botDragPayload(keys: string[]): string {
  return JSON.stringify(keys)
}

/** Read the payload back on drop. Never throws: a foreign or malformed drop
 *  yields no keys and the drop is simply ignored. */
export function readBotDragPayload(raw: string): string[] {
  try {
    const parsed: unknown = JSON.parse(raw)

    return Array.isArray(parsed) ? parsed.filter((k): k is string => typeof k === 'string' && Boolean(k)) : []
  } catch {
    return []
  }
}

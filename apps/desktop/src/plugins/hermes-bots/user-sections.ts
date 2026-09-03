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
import { botRosterMeta } from './routing'
import { getPluginCtx } from './shared'
import type { BotMeta, RosterRow } from './types'

export const UNASSIGNED_SECTION_KEY = 'section:unassigned'
export const BOT_SECTIONS_KEY = 'bot-sections-v1'

export interface BotSection {
  id: string
  name: string
}

/** `[{ id, name }]`, in display order. */
export const $botSections = atom<BotSection[]>([])

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

    if (!id || !name || seen.has(id)) {
      continue
    }

    seen.add(id)
    out.push({ id, name })
  }

  return out
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

/** Create a section and file `bots` into it. Returns the new section, or
 *  null when the name is blank. */
export function createBotSection(name: string, bots: RosterRow[] = []): BotSection | null {
  const clean = String(name || '').trim()

  if (!clean) {
    return null
  }

  const section: BotSection = { id: newSectionId(), name: clean }

  persistBotSections([...$botSections.get(), section])
  void moveBotsToSection(bots, section.id)

  return section
}

export function renameBotSection(id: string, name: string): void {
  const clean = String(name || '').trim()

  if (!clean) {
    return
  }

  persistBotSections($botSections.get().map(s => (s.id === id ? { ...s, name: clean } : s)))
}

/**
 * Delete the section only. Its members are not deleted and not hidden — they
 * fall back to Unassigned, which is the whole reason membership lives on the
 * bot rather than on the section. Returns an undo that puts the section back
 * in its slot and refiles the same bots, so the delete needs no confirmation.
 */
export function deleteBotSection(id: string, roster: RosterRow[] = []): { members: RosterRow[]; undo: () => void } {
  const list = $botSections.get()
  const index = list.findIndex(s => s.id === id)
  const section = list[index]
  const members = (roster || []).filter(bot => botSectionId(bot, $botMeta.get()) === id)

  persistBotSections(list.filter(s => s.id !== id))
  void moveBotsToSection(members, null)

  return {
    members,
    undo: () => {
      if (!section) {
        return
      }

      const current = $botSections.get().filter(s => s.id !== id)

      current.splice(Math.min(index, current.length), 0, section)
      persistBotSections(current)
      void moveBotsToSection(members, id)
    }
  }
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
  persistBotSections(next)
}

/**
 * `null` clears the assignment (back to Unassigned). One `saveBotMeta` per
 * bot — membership is a field on each bot's own profile, so that IS one write
 * per profile — and the writes run in sequence rather than fanned out, so the
 * shared local snapshot is never committed by two saves at once.
 */
export async function moveBotsToSection(bots: RosterRow[], sectionId: null | string): Promise<void> {
  for (const bot of bots || []) {
    if (bot && botSectionId(bot, $botMeta.get()) !== (sectionId || null)) {
      await saveBotMeta(bot, { sectionId: sectionId || null })
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

  blocks.push({ id: null, key: UNASSIGNED_SECTION_KEY, name: '', rows: loose })

  return blocks
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

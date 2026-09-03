/**
 * Which roster sections the user has folded away, and the storage round-trip
 * that makes that survive a restart.
 *
 * A leaf, like bot-state: the roster pane reads and writes it, and it reads
 * nothing back. The parse/serialize pair is pure so the persistence contract
 * can be tested without a plugin context or a rendered pane.
 *
 * Sections are identified by the same ids the pane already uses for its
 * headings — `team`, `group-chats`, and `gateway:<id>` — so a section that is
 * folded on one launch is folded on the next without the pane having to
 * enumerate them anywhere else.
 */

import { atom } from '@hermes/plugin-sdk'

import { getPluginCtx } from './shared'

export const ROSTER_COLLAPSE_KEY = 'roster-collapsed-sections-v1'

/** Section ids currently folded. Empty until `register()` hydrates it, so a
 *  window with no stored preference opens with everything expanded. */
export const $collapsedRosterSections = atom<Set<string>>(new Set())
/** Mirrors the selection-hydration flag: false until storage has answered,
 *  so the pane can avoid writing a default over a real stored value. */
export const $collapsedRosterSectionsHydrated = atom(false)

/**
 * Stored form is a plain string array — the smallest thing that survives
 * JSON round-tripping through plugin storage. Anything that is not an array of
 * non-empty strings is treated as absent rather than throwing: a corrupted or
 * hand-edited value must degrade to "nothing collapsed", never to a pane that
 * fails to render.
 */
export function parseCollapsedSections(value: unknown): Set<string> {
  if (!Array.isArray(value)) {
    return new Set()
  }

  const ids = value.filter((entry): entry is string => typeof entry === 'string' && entry.trim().length > 0)

  return new Set(ids.map(id => id.trim()))
}

/** Serialize for storage. Sorted so an unchanged selection produces an
 *  unchanged value and diffing a stored profile stays readable. */
export function serializeCollapsedSections(sections: Set<string>): string[] {
  return [...sections].sort()
}

/** Toggle one section id, returning a NEW set — the atom's subscribers only
 *  re-render on identity change. */
export function toggleCollapsedSection(sections: Set<string>, id: string): Set<string> {
  const next = new Set(sections)

  if (next.has(id)) {
    next.delete(id)
  } else {
    next.add(id)
  }

  return next
}

/** Write-through. Never rejects: storage being unavailable degrades to
 *  "collapse lasts for this window", exactly like the roster selection. */
export function persistCollapsedSections(sections: Set<string>): void {
  try {
    Promise.resolve(
      getPluginCtx()?.storage?.set?.(ROSTER_COLLAPSE_KEY, serializeCollapsedSections(sections))
    ).catch(() => undefined)
  } catch {
    /* storage unavailable — collapse lasts for this window */
  }
}

/** Toggle and persist in one step, so no caller can update the atom and
 *  forget the write. */
export function setSectionCollapsed(id: string): void {
  const next = toggleCollapsedSection($collapsedRosterSections.get(), id)

  $collapsedRosterSections.set(next)
  persistCollapsedSections(next)
}

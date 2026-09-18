/**
 * Session stamps — up to three short durable labels per session ("Merged", "WIP",
 * "Review", "Handoff", "Hold", or the user's own words), so a long session list
 * stays scannable instead of being a wall of similar titles.
 *
 * Unlike the sidebar's pins (localStorage with a backend mirror, see
 * session-pin-sync), a stamp has NO local copy: `sessions.stamps` on the
 * gateway's state.db is the truth, because it has to agree across two Desktop
 * installs and the CLI. So this module owns the write, patches the cached rows
 * optimistically, and puts the row back when the backend refuses — a stamp that
 * isn't on the server must never sit in the list looking durable.
 */

import { atom, computed, type WritableAtom } from 'nanostores'

import { setSessionStampsRemote } from '@/hermes'
import { Codecs, persistentAtom } from '@/lib/persisted'
import { readJson, writeJson } from '@/lib/storage'
import { notifyError } from '@/store/notifications'
import { $archivedSessions } from '@/store/sidebar-archive'
import type { SessionInfo } from '@/types/hermes'

import { $cronSessions, $messagingSessions, $sessions, sessionMatchesStoredId } from './session'

/** The menu's one-tap labels, in the order they are offered. Any other text the
 *  user types is equally valid — these are shortcuts, not an enum. */
export const SESSION_STAMP_PRESETS = ['Merged', 'WIP', 'Review', 'Handoff', 'Hold'] as const

/**
 * The Emoji panel's one-tap grid, shown before anything is typed. A curation,
 * not a whitelist: the panel searches the WHOLE bundled emoji catalog, and any
 * emoji reached that way becomes a menu title of its own (see `addStampTitle`).
 * These are just the ones a session list wants most often, one tap away with no
 * typing: state (🔥 in flight, ✅ done, 🚧 blocked, 💤 parked, ⏳ waiting), role
 * (👀 review, 🐛 bug, 📦 packaging, 🔍 investigating, 🔒 security), and outcome
 * (🚀 shipped, 🏁 finished, 🥇 best).
 */
export const SESSION_STAMP_EMOJI = [
  '🔥', '✅', '🚧', '👀', '🐛', '💤', '⏳', '🚀',
  '🎯', '⚠️', '🧪', '📌', '💡', '🧹', '❄️', '⭐',
  '🛠️', '📦', '🔍', '💬', '🧠', '🏁', '🔒', '📝'
] as const

/** How many search hits the Emoji panel paints at once. The grid scrolls, so
 *  this is a paint budget: enough to recognize a target by sight, few enough
 *  that a one-letter query does not mount the whole catalog. */
export const SESSION_STAMP_EMOJI_SEARCH_LIMIT = 48

/**
 * The Stamp submenu's EDITABLE title list, per CLIENT (localStorage — like the
 * sidebar's density). Two sides: `added` (titles the user created, in order) and
 * `deleted` (titles they took off the menu, stock or their own).
 *
 * Deleting or adding a title is a menu preference, never a data change: a session
 * already stamped "Hold" keeps reading "Hold" everywhere, it simply stops being
 * offered. `deleted` is stored rather than the surviving list so a preset added
 * to `SESSION_STAMP_PRESETS` later still shows up for everyone who never took it
 * off.
 */
const TITLES_KEY = 'hermes.desktop.sessionStampTitles.v1'

interface StampTitlePrefs {
  /** The user's own titles, in the order they added them. */
  added: string[]
  /** Titles taken off the menu, by label (restorable as a set). */
  deleted: string[]
}

const titleStrings = (value: unknown): string[] =>
  Array.isArray(value) ? value.filter((entry): entry is string => typeof entry === 'string') : []

function loadTitlePrefs(): StampTitlePrefs {
  const parsed = readJson<Partial<StampTitlePrefs>>(TITLES_KEY)

  return parsed && typeof parsed === 'object'
    ? { added: titleStrings(parsed.added), deleted: titleStrings(parsed.deleted) }
    : { added: [], deleted: [] }
}

export const $stampTitlePrefs = atom<StampTitlePrefs>(loadTitlePrefs())

/** The titles taken off the menu — the restore row shows while any is here. */
export const $deletedStampPresets = computed([$stampTitlePrefs], prefs => prefs.deleted)

/** Every title the Stamp submenu offers, in order: the stock presets the user
 *  kept, then the ones they added.
 *
 *  Keyed by LOWERCASE label, so one label is one row however it is spelled: a
 *  title the user added takes the slot of a stock preset with the same name
 *  (their spelling is the one on screen — retyping "hold" after taking "Hold"
 *  off the menu shows the row they just typed, not the old one), and a title
 *  they took off is gone for whatever else names it. */
export const $stampPresets = computed([$stampTitlePrefs], prefs => {
  const rows = new Map<string, string>()

  for (const preset of SESSION_STAMP_PRESETS) {
    rows.set(preset.toLowerCase(), preset)
  }

  for (const title of prefs.added) {
    rows.set(title.toLowerCase(), title)
  }

  for (const gone of prefs.deleted) {
    rows.delete(gone.toLowerCase())
  }

  return [...rows.values()]
})

const sameTitle = (one: string, other: string) => one.toLowerCase() === other.toLowerCase()

function saveTitlePrefs(prefs: StampTitlePrefs): void {
  $stampTitlePrefs.set(prefs)

  // Nothing custom left writes null rather than an empty object, so an untouched
  // install carries no key at all.
  writeJson(TITLES_KEY, prefs.added.length || prefs.deleted.length ? prefs : null)
}

/**
 * Add a title to the Stamp submenu, where it stays for the next session too.
 * A title already on the menu is a no-op. A title the user had REMOVED is added
 * as they just typed it: the removal marker goes with it, so the menu shows
 * their spelling instead of resurrecting the row they deleted.
 */
export function addStampTitle(label: string): void {
  const target = normalizeSessionStamp(label)

  if (!target) {
    return
  }

  const prefs = $stampTitlePrefs.get()

  if ($stampPresets.get().some(title => sameTitle(title, target))) {
    return
  }

  saveTitlePrefs({
    // At the END of their own titles, minus any stale copy of this label — a
    // title that was taken off and typed again is ONE entry, not two.
    added: [...prefs.added.filter(title => !sameTitle(title, target)), target],
    deleted: prefs.deleted.filter(title => !sameTitle(title, target))
  })
}

/** Take one title off the Stamp submenu. Idempotent, case-insensitive. */
export function deleteStampPreset(label: string): void {
  const target = normalizeSessionStamp(label)

  if (!target || $stampTitlePrefs.get().deleted.some(title => sameTitle(title, target))) {
    return
  }

  saveTitlePrefs({ ...$stampTitlePrefs.get(), deleted: [...$stampTitlePrefs.get().deleted, target] })
}

/** Put every taken-off title back (the row that shows while any is off). */
export function restoreStampPresets(): void {
  saveTitlePrefs({ ...$stampTitlePrefs.get(), deleted: [] })
}

/**
 * A stamp's COLOR per title, keyed by lowercased label — the same shape and home
 * as the per-session colors (store/session-color): desktop-local, one resolver
 * for every surface so the chip in the list and the chip in a tab can never
 * disagree. Setting null clears it and the chip falls back to the preset hue.
 */
export const $stampColorOverrides = persistentAtom<Record<string, string>>(
  'hermes.desktop.sessionStampColors',
  {},
  Codecs.stringRecord
)

const colorKey = (label: null | string | undefined): null | string => normalizeSessionStamp(label)?.toLowerCase() ?? null

/** The title's own color, or null to wear the label's default hue. */
export function stampColorFor(label: null | string | undefined, overrides: Record<string, string>): null | string {
  const key = colorKey(label)

  return key ? overrides[key] ?? null : null
}

/** Set (or clear, with null) one title's color. */
export function setStampColor(label: string, color: null | string): void {
  const key = colorKey(label)

  if (!key) {
    return
  }

  const previous = $stampColorOverrides.get()

  if (color) {
    $stampColorOverrides.set({ ...previous, [key]: color })

    return
  }

  if (key in previous) {
    const next = { ...previous }

    delete next[key]
    $stampColorOverrides.set(next)
  }
}

/** The stamp color panel's swatches: FINER than the profile rail's twelve
 *  (lib/profile-color), because two stamps share one screen and 30° steps made
 *  "red" and "orange" the same pick. Same saturation/lightness as that palette,
 *  so a stamp still wears an app color, and every one of its hues sits on this
 *  wheel at the same value — a color picked before this existed still shows its
 *  own swatch as the current one. */
export const STAMP_SWATCHES: readonly string[] = Array.from(
  { length: 24 },
  (_, index) => `hsl(${index * 15} 68% 58%)`
)

/** Cap on a stamp's length: long enough for "Waiting on CI", short enough to
 *  stay a label beside a title. Mirrors the backend's own limit, which rejects
 *  anything longer, so the UI never sends what the API would 400. */
export const SESSION_STAMP_MAX_LENGTH = 24

/** How many labels one session may carry — mirrors the backend's cap, because a
 *  list row and a tab have to stay readable. The menu stops offering more at the
 *  limit instead of letting the API refuse a click the user cannot explain. */
export const SESSION_STAMP_LIMIT = 3

/** Trim, collapse whitespace runs, cap the length. `''`, `null` and `undefined`
 *  all mean "no stamp". The ONE normalizer shared by the writer, the chip and
 *  the menu's custom-text input.
 *
 *  The cap counts CODE POINTS, not UTF-16 units: a stamp may legitimately be an
 *  emoji (👨‍👩‍👧 is five code points, eight units), and a plain `slice(0, 24)`
 *  cuts the pair in half when the cut lands inside one — the backend then stores
 *  a lone surrogate and the chip paints a replacement glyph. Cutting on points
 *  can only ever drop a whole character. */
export function normalizeSessionStamp(raw: null | string | undefined): null | string {
  const value = (raw ?? '').trim().replace(/\s+/g, ' ')

  if (!value) {
    return null
  }

  const points = Array.from(value)

  return points.length > SESSION_STAMP_MAX_LENGTH ? points.slice(0, SESSION_STAMP_MAX_LENGTH).join('') : value
}

/**
 * A stamp that is nothing but emoji (one glyph, a ZWJ sequence like 👨‍👩‍👧, a
 * skin-toned 👍🏽, a flag, or several of them in a row).
 *
 * It matters because an emoji is drawn by the platform's colour emoji font,
 * which a CSS `color` cannot tint — so an emoji stamp paints as the glyph itself
 * (larger, no wash) and the menu offers it no colour choice. The test requires
 * at least one Extended_Pictographic, so a plain digit or `#` never reads as
 * emoji; the tail covers the joiners, variation selectors, keycap marks and skin
 * tones that make one emoji out of several points, and flags are their own
 * regional-indicator pair.
 *
 * Written as an alternation rather than a character class: a class holding a ZWJ
 * or a variation selector is exactly what `no-misleading-character-class` exists
 * to catch, and spelling the points out says what is allowed.
 */
const EMOJI_TAIL = '(?:\\u200D|\\uFE0F|\\u20E3|\\u{1F3FB}|\\u{1F3FC}|\\u{1F3FD}|\\u{1F3FE}|\\u{1F3FF})*'
const EMOJI_STAMP_RE = new RegExp(`^(?:\\p{Extended_Pictographic}${EMOJI_TAIL})+$`, 'u')
const FLAG_STAMP_RE = /^\p{Regional_Indicator}{2}$/u

export function isEmojiStamp(label: null | string | undefined): boolean {
  const value = normalizeSessionStamp(label)

  if (!value) {
    return false
  }

  return EMOJI_STAMP_RE.test(value) || FLAG_STAMP_RE.test(value)
}

/**
 * The labels a row carries, in order.
 *
 * The backend projects `stamps` as a real list; `stamp` is the single label an
 * older backend (or a row the user just stamped) leaves behind — so this ONE
 * helper is where "one label" reads as "a list of one". Every surface goes
 * through it rather than testing `row.stamps` in one place and `row.stamp` in
 * another, which is how the list and the tabs drift apart.
 */
export function stampLabels(row: null | Pick<SessionInfo, 'stamp' | 'stamps'> | undefined): string[] {
  const labels = normalizeSessionStamps(row?.stamps ?? [])

  // Read-compat: a row carrying only the singular label (an older backend, or an
  // optimistic write before the list landed) is a session with one stamp.
  return labels.length ? labels : normalizeSessionStamps([row?.stamp])
}

/** Normalize a whole list the way the writer stores it: each label through the
 *  ONE normalizer, blanks dropped, case-insensitive duplicates collapsed onto
 *  their first spelling, order preserved. */
export function normalizeSessionStamps(labels: readonly (null | string | undefined)[]): string[] {
  const out: string[] = []

  for (const raw of labels) {
    const label = normalizeSessionStamp(raw)

    if (label && !out.some(seen => sameTitle(seen, label))) {
      out.push(label)
    }
  }

  return out
}

/** True when *label* is one of *labels* — case-insensitively, because "mrg" and
 *  "MRG" are the same stamp. */
export function hasStampLabel(labels: readonly string[], label: null | string | undefined): boolean {
  const target = normalizeSessionStamp(label)

  return Boolean(target) && labels.some(seen => sameTitle(seen, target as string))
}

/** Every atom that can hold a session row the user might be looking at. */
const stampAtoms: WritableAtom<SessionInfo[]>[] = [
  $sessions,
  $cronSessions,
  $messagingSessions,
  $archivedSessions
]

/** Value-equality for two label lists: same labels, same order. */
const sameLabels = (one: readonly string[], other: readonly string[]): boolean =>
  one.length === other.length && one.every((label, index) => label === other[index])

/** Write *labels* onto a row, keeping the singular mirror in step so a cached row
 *  never disagrees with itself. */
const withStamps = (row: SessionInfo, labels: string[]): SessionInfo => ({
  ...row,
  stamp: labels[0] ?? null,
  stamps: labels
})

function patchRow(rows: SessionInfo[], sessionId: string, labels: string[]): SessionInfo[] {
  let mutated = false

  const next = rows.map(row => {
    // Lineage-aware, like every READER of this id (`sessionMatchesStoredId`). A
    // tab's stored id is the one the tile was opened with — regularly the
    // lineage ROOT by the time a conversation has compressed — so an `id ===`
    // test here painted nothing and the stamp only showed up on the next list
    // poll, which is what "sometimes it doesn't reach the tab" was.
    if (!sessionMatchesStoredId(row, sessionId) || sameLabels(stampLabels(row), labels)) {
      return row
    }

    mutated = true

    return withStamps(row, labels)
  })

  // Preserve reference identity on a no-op: handing React a fresh array with the
  // same rows re-renders the whole expensive tree for nothing.
  return mutated ? next : rows
}

/** Roll one row back to *labels* in every list, without clobbering the rows a
 *  concurrent poll may have replaced meanwhile. */
function restoreRow(sessionId: string, labels: string[]): void {
  for (const atom of stampAtoms) {
    atom.set(patchRow(atom.get(), sessionId, labels))
  }
}

/** The pre-write labels of the row, for the rollback path. */
function currentStamps(sessionId: string): string[] {
  for (const atom of stampAtoms) {
    const row = atom.get().find(candidate => sessionMatchesStoredId(candidate, sessionId))

    if (row) {
      return stampLabels(row)
    }
  }

  return []
}

/**
 * Replace a session's whole stamp list, optimistically. Resolves to whether the
 * backend accepted it — callers rarely care, but the failure path is visible:
 * the row snaps back and a notification says why.
 */
export async function applySessionStamps(
  sessionId: string,
  profile: string | undefined,
  labels: readonly (null | string | undefined)[]
): Promise<boolean> {
  const next = normalizeSessionStamps(labels)
  const previous = currentStamps(sessionId)

  restoreRow(sessionId, next)

  try {
    await setSessionStampsRemote(sessionId, next, profile)

    return true
  } catch (error) {
    restoreRow(sessionId, previous)
    notifyError(error, 'Could not set the session stamps')

    return false
  }
}

/**
 * Set (or clear) one session's stamp. The single-label door: it REPLACES the
 * list, which is what "stamp the session with X" has always meant and what the
 * CLI's `sessions stamp` does with the same store.
 */
export async function applySessionStamp(
  sessionId: string,
  profile: string | undefined,
  stamp: null | string
): Promise<boolean> {
  return applySessionStamps(sessionId, profile, [stamp])
}

/**
 * Toggle one label on a session — the Stamp submenu's row action: absent adds it
 * at the END (order = the order the user built it in), present lifts it off and
 * leaves the remaining labels where they are.
 *
 * At :data:`SESSION_STAMP_LIMIT` labels an ABSENT label is refused here, without
 * a write and without a notification: the menu shows the limit, so a click that
 * cannot land must not look like one that did.
 */
export async function toggleSessionStamp(
  sessionId: string,
  profile: string | undefined,
  label: null | string
): Promise<boolean> {
  const target = normalizeSessionStamp(label)

  if (!target) {
    return false
  }

  const current = currentStamps(sessionId)
  const present = hasStampLabel(current, target)

  if (!present && current.length >= SESSION_STAMP_LIMIT) {
    return false
  }

  return applySessionStamps(
    sessionId,
    profile,
    present ? current.filter(seen => !sameTitle(seen, target)) : [...current, target]
  )
}

/**
 * Stored/lineage id -> stamp LABELS, over every loaded row.
 *
 * The tab strip subscribes to this through a string selector rather than to the
 * session lists themselves: the map is rebuilt once per list poll (O(rows),
 * once), while a selector keyed on ITS OWN labels ignores the churn — so a tab
 * repaints when its stamps change, not on every poll. Same shape as
 * `$sessionDotStateById`.
 */
export const $sessionStamps = computed(
  [$sessions, $cronSessions, $messagingSessions],
  (rows, cronRows, messagingRows) => {
    const stamps = new Map<string, string[]>()

    for (const session of [...rows, ...cronRows, ...messagingRows]) {
      const labels = stampLabels(session)

      if (!labels.length) {
        continue
      }

      // Compression moves a conversation to a new row id, so a pane or a page
      // holding an older id must still resolve the stamps its session carries.
      stamps.set(session.id, labels)

      if (session._lineage_root_id) {
        stamps.set(session._lineage_root_id, labels)
      }

      for (const id of session._lineage_ids ?? []) {
        stamps.set(id, labels)
      }
    }

    return stamps
  }
)

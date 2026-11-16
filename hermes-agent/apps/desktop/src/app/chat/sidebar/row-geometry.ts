import { cn } from '@/lib/utils'

// The sidebar row's measurements, in one place. The session row is canonical
// and everything else composes these — a row that spells its own geometry
// floats 1–2px off the sessions above it, which is how the drift starts.
//
// Split out of `chrome.tsx` so the literals are reachable without the
// components: light, store-free callers (a connection glyph) and the plugin
// SDK both need the lead cell, and neither should have to pull nanostores to
// line up with a session row.
//
// Height lives ONLY on the shell (`SIDEBAR_ROW_MIN_H`). Inset children stretch
// to fill the cell and center content internally — never `items-center` on the
// shell grid, or short clusters (projects) float off sessions.
//
// `SIDEBAR_ROW_PAD_X` is the BODY's padding: the lead's inset, plus the gap the
// label keeps from the actions column, both inside the row's click target.
// `SIDEBAR_ROW_PAD_TRAIL` is the row's own trailing inset and belongs to the
// SHELL — the only box containing both the actions column AND the card's
// in-body cluster, so one class insets every trailing thing a row can render.
// Owned anywhere else, the age / chips / kebab sit flush on the border box,
// which is exactly where a working row paints its arc (`.arc-row` has zero
// standoff) — the ring ran through the text.

export const SIDEBAR_ROW_MIN_H = 'min-h-[1.625rem]' as const
export const SIDEBAR_ROW_PAD_X = 'pl-2 pr-2' as const
export const SIDEBAR_ROW_PAD_TRAIL = 'pr-2' as const
export const SIDEBAR_ROW_GAP = 'gap-1.5' as const

/** Fixed leading cell — dot, icon, drag handle. Every row's label starts at the
 *  same left edge because they all reserve this exact box. */
export const SIDEBAR_ROW_LEAD = 'grid size-3.5 shrink-0 place-items-center' as const

export const SIDEBAR_ROW_INSET = cn(
  SIDEBAR_ROW_PAD_X,
  SIDEBAR_ROW_GAP,
  'flex h-full min-w-0 items-center self-stretch py-0.5'
)

// `truncate` is overflow:hidden. `leading-none` (line-height: 1) makes the line
// box equal the em-square, so glyph ink that sticks out — Segoe UI on Windows
// is ~1.33em — gets shaved. 1.35 leaves room; the shell still owns row height,
// so the extra leading just centers.
export const SIDEBAR_TRUNCATED_LEADING = 'leading-[1.35]' as const

export const SIDEBAR_ROW_LABEL = cn(
  'min-w-0 truncate text-[0.8125rem] text-(--ui-text-secondary)',
  SIDEBAR_TRUNCATED_LEADING
)

/** Inbox-style card (workspace + age, title + preview, model + size). */
export const SIDEBAR_ROW_CARD_MIN_H = 'min-h-[3.375rem]' as const

/** Codicon size in sidebar row leads — matches the file tree (`tree.tsx`). */
export const SIDEBAR_LEAD_ICON_SIZE = '0.875rem' as const

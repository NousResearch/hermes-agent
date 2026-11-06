// Pure timeline helpers — no React/DOM; tested in thread-timeline-data.test.ts.
import { PROCESS_NOTIFICATION_RE } from './content'

export interface TimelineSourceMessage {
  id: string
  rowId?: number
  role: string
  text: string
}

export interface TimelineEntry {
  id: string
  rowId?: number
  preview: string
  /** First characters, for the Dock label beside the magnified tick. */
  label: string
}

const PREVIEW_MAX = 120

/** The box beside the rail replaces the tick, so it has to identify a prompt
 *  at a glance without a tooltip: the full prompt never fits, and 30 characters
 *  is where a Japanese prompt still says what it was about. */
const LABEL_MAX = 30

/** Rows away from the pointer where the Dock ramp has decayed to zero. Three
 *  neighbours rising as the cursor moves is what makes it read as a launcher
 *  ramp instead of a single hovered row. */
const DOCK_RADIUS = 2.6

/**
 * How much a mark is magnified, by its distance from the pointer in rows: 1
 * under the cursor, 0 at `radius`. A raised cosine, so it is continuous — the
 * marks follow the cursor between ticks instead of snapping to the nearest one.
 */
export function dockScale(distance: number, radius: number = DOCK_RADIUS): number {
  const clamped = Math.abs(distance) / radius

  return !Number.isFinite(clamped) || clamped >= 1 ? 0 : 0.5 + 0.5 * Math.cos(Math.PI * clamped)
}

/** A row as it is painted: magnified, and pushed down by the growth above it. */
export interface TimelineDockRow {
  /** Row top in track coordinates. */
  top: number
  /** Row height; `pitch` when the pointer is far away. */
  height: number
  /** Magnification 0..1, drives the mark's size. */
  scale: number
}

/**
 * macOS-Dock magnification for a vertical rail. `anchor` is the pointer in track
 * coordinates (null when the pointer is elsewhere) and `grow` the extra pixels a
 * fully magnified row gains. Each row grows around its own centre and every row
 * below a magnified one is pushed down by the growth above it, which is what
 * makes the neighbours slide aside instead of overlapping.
 *
 * The caller MUST give the rail the matching extra height (see `dockGrowth`) —
 * rows are not allowed to leave the rail just because the cursor is on it.
 */
export function dockRows(
  starts: readonly number[],
  pitch: number,
  anchor: number | null,
  grow: number,
  radius: number = DOCK_RADIUS
): TimelineDockRow[] {
  if (anchor === null || !Number.isFinite(anchor)) {
    return starts.map(start => ({ height: pitch, scale: 0, top: start }))
  }

  let pushed = 0

  return starts.map(start => {
    const scale = dockScale((anchor - (start + pitch / 2)) / pitch, radius)
    const growth = grow * scale
    const row = { height: pitch + growth, scale, top: start + pushed - growth / 2 }

    pushed += growth

    return row
  })
}

/** Total height the painted rows add over the resting pitch. */
export function dockGrowth(rows: readonly TimelineDockRow[], pitch: number): number {
  return rows.reduce((sum, row) => sum + (row.height - pitch), 0)
}

/** A painted row's hit band. */
export interface TimelineBand {
  index: number
  top: number
  height: number
}

/**
 * Which prompt owns a point in track coordinates, read off the PAINTED bands.
 * A magnified row is ~3x the resting pitch, so hit-testing the uniform grid
 * picks the neighbour while the cursor is still inside the selected row's own
 * band. The current selection keeps the point while it stays inside its band,
 * which is what makes the band stable instead of flickering at the edges.
 */
export function timelineIndexAt(
  y: number,
  bands: readonly TimelineBand[],
  pitch: number,
  count: number,
  current: number | null = null
): number {
  const contains = (band: TimelineBand) => y >= band.top && y < band.top + band.height

  if (current !== null) {
    for (const band of bands) {
      if (band.index === current && contains(band)) {
        return current
      }
    }
  }

  for (const band of bands) {
    if (contains(band)) {
      return band.index
    }
  }

  return Math.max(0, Math.min(count - 1, Math.round(y / pitch - 0.5)))
}

/** Shared, localized focus curve; the active tick always retains its maximum. */
export function timelineBarWidth(index: number, activeIndex: number, hoverIndex: number | null): number {
  const curve = (distance: number) => (Math.abs(distance) > 3 ? 0 : Math.exp(-(distance ** 2) / 1.35))

  return 0.5 + 0.5 * Math.max(curve(index - activeIndex), hoverIndex === null ? 0 : curve(index - hoverIndex))
}

export const TIMELINE_REVEAL_EVENT = 'hermes:timeline-reveal'
export const EARLIER_TIMELINE_ID = '__earlier-history__'

export interface TimelineRevealRequest {
  id: string
  rowId?: number
  signal: AbortSignal
  complete: (revealedId: string | false) => void
}

export function timelinePreview(text: string, max: number = PREVIEW_MAX): string {
  const collapsed = text.replace(/\s+/g, ' ').trim()

  if (collapsed.length <= max) {
    return collapsed
  }

  return `${collapsed.slice(0, max - 1).trimEnd()}…`
}

/** The short label. Counts code points, not UTF-16 units, so a CJK or emoji
 *  prompt is not cut mid-character into a replacement glyph. */
export function timelineDockLabel(text: string, max: number = LABEL_MAX): string {
  const collapsed = text.replace(/\s+/g, ' ').trim()
  const chars = Array.from(collapsed)

  if (chars.length <= max) {
    return collapsed
  }

  return `${chars.slice(0, max).join('').trimEnd()}…`
}

export function deriveTimelineEntries(messages: readonly TimelineSourceMessage[]): TimelineEntry[] {
  const entries: TimelineEntry[] = []

  for (const message of messages) {
    if (message.role !== 'user') {
      continue
    }

    const text = message.text.trim()

    if (!text || PROCESS_NOTIFICATION_RE.test(text)) {
      continue
    }

    entries.push({
      id: message.id,
      label: timelineDockLabel(text),
      preview: timelinePreview(text),
      ...(message.rowId !== undefined ? { rowId: message.rowId } : {})
    })
  }

  return entries
}

/** Do two derivations describe the same rail? Lets a rebuild hand back the
 *  PREVIOUS array so an unchanged transcript costs zero re-renders. */
export function sameTimelineEntries(a: readonly TimelineEntry[], b: readonly TimelineEntry[]): boolean {
  if (a === b) {
    return true
  }

  if (a.length !== b.length) {
    return false
  }

  return a.every(
    (entry, index) =>
      entry.id === b[index].id && entry.preview === b[index].preview && entry.label === b[index].label
  )
}

/** Last user prompt at/above the viewport top (with slack); else first rendered. */
export function activeTimelineIndex(offsets: readonly (number | null)[], slack: number = 8): number {
  let active = -1
  let firstRendered = -1

  for (let i = 0; i < offsets.length; i++) {
    const offset = offsets[i]

    if (offset == null) {
      continue
    }

    if (firstRendered === -1) {
      firstRendered = i
    }

    if (offset <= slack) {
      active = i
    }
  }

  if (active !== -1) {
    return active
  }

  return firstRendered === -1 ? 0 : firstRendered
}

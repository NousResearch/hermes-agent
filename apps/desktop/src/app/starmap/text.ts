import { translateNow, type Translations } from '@/i18n'
import { dateFor } from '@/lib/time'
import type { StarmapNode } from '@/types/hermes'

// Resolved lazily, never as a default argument: the ring-label loops call
// formatDate once per ring and only the undated ones need the fallback.
// English resolves to the bare `unknown` this function has always returned.
function unknownDate(supplied?: string): string {
  return supplied ?? translateNow('starmap.unknown')
}

export function formatDate(ts: null | number | undefined, locale: string, unknown?: string): string {
  if (!ts) {
    return unknownDate(unknown)
  }

  try {
    return dateFor(locale).format(new Date(ts * 1000))
  } catch {
    return unknownDate(unknown)
  }
}

// Tag-style badge items for the hover tooltip — date first. Use-count is NOT a
// badge (rendered separately, right-aligned) so it's excluded here.
export function metaBadges(n: StarmapNode, t: Translations, locale: string): string[] {
  const out: string[] = [formatDate(n.timestamp, locale, t.starmap.unknown)]

  if (n.kind === 'memory') {
    out.push(n.memorySource === 'profile' ? t.starmap.profileMemory : t.starmap.memoryBadge)
  } else {
    // The raw backend slug is the badge, exactly as it always was — prettifying
    // it here would rewrite `smart-home` into `Smart Home` for English users.
    // Arabic has no readable fallback for a slug, so it alone maps the label.
    out.push((locale === 'ar' && t.skills.categoryLabels?.[n.category.trim().toLowerCase()]) || n.category)

    if (n.createdBy === 'agent') {
      out.push(t.starmap.learned)
    }

    if (n.pinned) {
      out.push(t.starmap.pinned)
    }
  }

  return out.filter(Boolean)
}

// Bare "xN" use-count, last in the badge row. Null when never used.
export function countLabel(n: StarmapNode): null | string {
  return n.kind === 'skill' && n.useCount > 0 ? `x${n.useCount}` : null
}

// Footer-row content for the tooltip. Reserved primitive — returns nothing for
// now (skills have no UUID; their id is just the name). Wire real detail here
// later and the tooltip lays it out automatically.
export function nodeFooter(node: StarmapNode): null | string {
  void node

  return null
}

// Greedy word-wrap for the tooltip title so long memory lines don't blow out.
export function wrapText(ctx: CanvasRenderingContext2D, text: string, maxW: number): string[] {
  const words = text.split(/\s+/).filter(Boolean)
  const lines: string[] = []
  let line = ''

  for (const word of words) {
    const next = line ? `${line} ${word}` : word

    if (!line || ctx.measureText(next).width <= maxW) {
      line = next
    } else {
      lines.push(line)
      line = word
    }
  }

  if (line) {
    lines.push(line)
  }

  return lines
}

// Trim to fit maxW, appending an ellipsis (keeps floating labels compact so they
// don't span the overlay).
export function ellipsize(ctx: CanvasRenderingContext2D, text: string, maxW: number): string {
  if (ctx.measureText(text).width <= maxW) {
    return text
  }

  let s = text

  while (s.length > 1 && ctx.measureText(`${s}…`).width > maxW) {
    s = s.slice(0, -1)
  }

  return `${s.trimEnd()}…`
}

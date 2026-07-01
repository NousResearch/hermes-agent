import type { StarmapNode } from '@/types/hermes'

export function formatDate(ts?: null | number): string {
  if (!ts) {
    return 'unknown'
  }

  try {
    return new Date(ts * 1000).toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })
  } catch {
    return 'unknown'
  }
}

// Tag-style badge items for the hover tooltip — date first. Use-count is NOT a
// badge (rendered separately, right-aligned) so it's excluded here.
export function metaBadges(n: StarmapNode): string[] {
  const out: string[] = [formatDate(n.timestamp)]

  if (n.kind === 'memory') {
    out.push(n.memorySource === 'profile' ? 'profile memory' : 'memory')
  } else {
    out.push(n.category)

    if (n.createdBy === 'agent') {
      out.push('learned')
    }

    if (n.pinned) {
      out.push('pinned')
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

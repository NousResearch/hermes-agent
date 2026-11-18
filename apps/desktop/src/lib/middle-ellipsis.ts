/**
 * The longest `head…tail` form of `text` that fits `width`, keeping the tail (a path's leaf) whole
 * when it can: `~/.cache/omp…/apps/desktop/portal`. One string, so nothing is left over between the
 * two ends the way two CSS-truncated spans leave a gap. `measure` returns a string's rendered width.
 */
export function middleEllipsis(text: string, width: number, measure: (value: string) => number): string {
  if (measure(text) <= width) {
    return text
  }

  // Binary search the number of characters kept; the tail gets the larger half, so the leaf
  // survives longest.
  let low = 0
  let high = text.length - 1
  let best = '…'

  while (low <= high) {
    const keep = (low + high) >> 1
    const tail = Math.ceil(keep * 0.6)
    const candidate = `${text.slice(0, keep - tail)}…${text.slice(text.length - tail)}`

    if (measure(candidate) <= width) {
      best = candidate
      low = keep + 1
    } else {
      high = keep - 1
    }
  }

  return best
}

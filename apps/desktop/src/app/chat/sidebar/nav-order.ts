export interface PositionedNavItem {
  id: string
  override?: string
  position?: string
  route?: string
}

/** Merge manifest-driven nav rows into the existing nav without hardcoding a
 * plugin. `before:<id>` / `after:<id>` may target either a core row or another
 * contributed row; unresolved anchors fall back to the end in registry order. */
export function mergePositionedNav<T extends PositionedNavItem>(core: readonly T[], contributed: readonly T[]): T[] {
  const merged = [...core]
  let pending = contributed.filter(item => {
    if (!item.override) {
      return true
    }

    const overrideIndex = merged.findIndex(candidate => candidate.route === item.override || candidate.id === item.override?.replace(/^\//, ''))

    if (overrideIndex < 0) {
      return true
    }

    merged.splice(overrideIndex, 1, item)

    return false
  })

  while (pending.length > 0) {
    const nextPending: T[] = []
    let inserted = 0

    for (const item of pending) {
      const parsed = parsePosition(item.position)

      if (!parsed) {
        merged.push(item)
        inserted += 1

        continue
      }

      const anchorIndex = merged.findIndex(candidate => candidate.id === parsed.anchor)

      if (anchorIndex < 0) {
        nextPending.push(item)

        continue
      }

      if (parsed.side === 'before') {
        merged.splice(anchorIndex, 0, item)
      } else {
        let insertAt = anchorIndex + 1

        while (insertAt < merged.length && merged[insertAt].position === item.position) {
          insertAt += 1
        }

        merged.splice(insertAt, 0, item)
      }

      inserted += 1
    }

    if (inserted === 0) {
      merged.push(...nextPending)

      break
    }

    pending = nextPending
  }

  return merged
}

function parsePosition(position: string | undefined): { anchor: string; side: 'after' | 'before' } | null {
  const match = position?.match(/^(after|before):([a-z0-9._-]+)$/i)

  return match ? { anchor: match[2], side: match[1] as 'after' | 'before' } : null
}

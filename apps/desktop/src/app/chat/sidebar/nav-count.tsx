import { useStore } from '@nanostores/react'
import type { ReadableAtom } from 'nanostores'

import { Badge } from '@/components/ui/badge'

/** `countLabel` is plugin code running inside the core sidebar's render, so a
 *  throw there falls back to the bare number instead of taking the sidebar
 *  down. (Not a `ContribBoundary`: its fallback is a button, and this badge
 *  already sits inside the row's button.) */
function safeCountLabel(countLabel: ((count: number) => string) | undefined, n: number): string {
  if (!countLabel) {
    return String(n)
  }

  try {
    return countLabel(n)
  } catch (error) {
    console.warn('[sidebar.nav] countLabel threw; using the bare count', error)

    return String(n)
  }
}

/** Trailing count on a contributed nav row. Subscribes to its own atom so a
 *  changing count re-renders this leaf only, never the sidebar. */
export function NavCount({
  count,
  countLabel
}: {
  count: ReadableAtom<number>
  countLabel?: (count: number) => string
}) {
  const n = useStore(count)

  if (n <= 0) {
    return null
  }

  return (
    <Badge aria-label={safeCountLabel(countLabel, n)} className="ml-auto tabular-nums" size="xs" variant="muted">
      {n > 99 ? '99+' : n}
    </Badge>
  )
}

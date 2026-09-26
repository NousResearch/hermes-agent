import { useStore } from '@nanostores/react'
import type { ReadableAtom } from 'nanostores'

import { Badge } from '@/components/ui/badge'

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
    <Badge aria-label={countLabel?.(n) ?? String(n)} className="ml-auto tabular-nums" size="xs" variant="muted">
      {n > 99 ? '99+' : n}
    </Badge>
  )
}

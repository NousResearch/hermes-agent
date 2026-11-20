import type { CSSProperties } from 'react'

import { Skeleton } from '@/components/ui/skeleton'

// Shared loading skeletons for the file/git trees and diffs — quieter than a
// spinner and shaped like the content that's about to land.

const TREE_ROWS: { indent: number; width: string }[] = [
  { indent: 0, width: '55%' },
  { indent: 1, width: '72%' },
  { indent: 1, width: '46%' },
  { indent: 0, width: '60%' },
  { indent: 1, width: '52%' },
  { indent: 2, width: '40%' },
  { indent: 0, width: '64%' }
]

/** Rows of icon + label bars, mimicking a file tree mid-load. */
export function TreeSkeleton() {
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2 px-3 py-2.5" data-slot="tree-skeleton">
      {TREE_ROWS.map((row, index) => (
        <div
          className="flex items-center gap-2"
          key={`${index}-${row.width}`}
          style={{ paddingLeft: `${row.indent * 12}px` }}
        >
          <Skeleton className="size-3.5 shrink-0 rounded-[3px]" />
          <Skeleton className="h-3" style={{ width: row.width }} />
        </div>
      ))}
    </div>
  )
}

const DIFF_ROWS: string[] = ['72%', '40%', '88%', '55%', '64%', '30%', '80%', '48%', '60%', '36%', '70%']

/** Stacked line bars, mimicking a unified diff mid-load. */
export function DiffSkeleton({ style }: { style?: CSSProperties }) {
  return (
    <div className="flex flex-col gap-1.5 px-3 py-2" data-slot="diff-skeleton" style={style}>
      {DIFF_ROWS.map((width, index) => (
        <Skeleton className="h-3" key={`${index}-${width}`} style={{ width }} />
      ))}
    </div>
  )
}

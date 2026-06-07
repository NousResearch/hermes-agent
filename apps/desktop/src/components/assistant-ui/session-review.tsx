import { useState } from 'react'
import { useStore } from '@nanostores/react'

import { Codicon } from '@/components/ui/codicon'
import { DiffLines } from '@/components/chat/diff-lines'
import { ChevronDown, ChevronRight, FileText } from '@/lib/icons'
import { $sessionChanges, stripAnsi } from '@/store/session-changes'

export function SessionReview() {
  const changes = useStore($sessionChanges)
  const [expanded, setExpanded] = useState<Set<number>>(new Set())

  // Only show entries with actual diffs
  const diffChanges = changes.filter(c => c.diff.trim().length > 0)

  const toggle = (index: number) => {
    setExpanded(prev => {
      const next = new Set(prev)
      if (next.has(index)) next.delete(index)
      else next.add(index)
      return next
    })
  }

  if (diffChanges.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-8 text-center text-xs text-muted-foreground">
        <Codicon name="diff" size={24} className="mb-2 opacity-40" />
        No file changes to review
      </div>
    )
  }

  return (
    <div className="flex flex-col overflow-y-auto text-xs">
      {/* Header */}
      <div className="sticky top-0 z-10 flex items-center gap-2 border-b border-border bg-background px-3 py-2">
        <Codicon name="diff" size={14} className="text-muted-foreground" />
        <span className="font-medium text-foreground">Review Changes</span>
        <span className="ml-auto text-muted-foreground">{diffChanges.length} files</span>
      </div>

      {/* File list */}
      {diffChanges.map((change, index) => {
        const isExpanded = expanded.has(index)
        const deletions = countDeletions(change.diff)

        return (
          <div key={index} className="border-b border-border">
            {/* File row */}
            <button
              className="flex w-full items-center gap-2 px-3 py-1.5 text-left hover:bg-muted/50"
              onClick={() => toggle(index)}
            >
              {isExpanded ? (
                <ChevronDown size={12} className="shrink-0 text-muted-foreground" />
              ) : (
                <ChevronRight size={12} className="shrink-0 text-muted-foreground" />
              )}
              <FileText size={12} className="shrink-0 text-muted-foreground" />
              <span className="truncate text-foreground">{shortPath(change.path)}</span>
              <span className="ml-auto shrink-0 text-emerald-500">+{change.additions}</span>
              <span className="shrink-0 text-rose-500">-{deletions}</span>
            </button>

            {/* Expanded diff */}
            {isExpanded && (
              <div className="border-t border-border bg-muted/20">
                <DiffLines text={stripAnsi(change.diff)} />
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

function shortPath(path: string): string {
  const parts = path.replace(/\\/g, '/').split('/')
  return parts.length > 2 ? `.../${parts.slice(-2).join('/')}` : path
}

function countDeletions(diff: string): number {
  let count = 0
  for (const line of diff.split('\n')) {
    if (line.startsWith('-') && !line.startsWith('---')) count++
  }
  return count
}

import * as React from 'react'

import { cn } from '@/lib/utils'

/**
 * Per-line classed renderer for unified diffs. Lives outside `CodeCard` so
 * tool-result panels (already nested inside a tool card) don't double-shell;
 * for markdown ` ```diff ` fences the standard `CodeCard` + Shiki path runs
 * instead and gives equivalent coloring.
 */
interface DiffLineKind {
  className?: string
  match: (line: string) => boolean
}

const DIFF_LINE_KINDS: DiffLineKind[] = [
  {
    className: 'text-emerald-700 dark:text-emerald-300',
    match: line => line.startsWith('+') && !line.startsWith('+++')
  },
  { className: 'text-rose-700 dark:text-rose-300', match: line => line.startsWith('-') && !line.startsWith('---') },
  { className: 'text-sky-700 dark:text-sky-300', match: line => line.startsWith('@@') },
  {
    className: 'text-muted-foreground/70',
    match: line => line.startsWith('---') || line.startsWith('+++') || / → /.test(line.slice(0, 60))
  }
]

function classifyLine(line: string): string | undefined {
  return DIFF_LINE_KINDS.find(kind => kind.match(line))?.className
}

interface DiffLinesProps extends Omit<React.ComponentProps<'pre'>, 'children'> {
  text: string
}

export function DiffLines({ className, text, ...props }: DiffLinesProps) {
  return (
    <pre
      className={cn(
        'mt-1 mb-1.5 max-h-96 max-w-full min-w-0 overflow-auto rounded-md border border-border/60 bg-muted/35 px-2.5 py-1.5 font-mono text-[0.7rem] leading-relaxed text-muted-foreground',
        className
      )}
      data-slot="diff-lines"
      {...props}
    >
      {text.split('\n').map((line, index) => (
        <span className={cn('block min-w-max whitespace-pre', classifyLine(line))} key={`${index}-${line}`}>
          {line || ' '}
        </span>
      ))}
    </pre>
  )
}

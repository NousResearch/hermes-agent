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
    className: 'border-emerald-500 bg-emerald-500/12 text-emerald-800 dark:text-emerald-200',
    match: line => line.startsWith('+') && !line.startsWith('+++')
  },
  {
    className: 'border-rose-500 bg-rose-500/12 text-rose-800 dark:text-rose-200',
    match: line => line.startsWith('-') && !line.startsWith('---')
  },
  {
    className: 'text-sky-700 dark:text-sky-300',
    match: line => line.startsWith('@@')
  },
  {
    className: 'text-muted-foreground/70',
    match: line => line.startsWith('---') || line.startsWith('+++') || / → /.test(line.slice(0, 60))
  }
]

function classifyLine(line: string): string | undefined {
  return DIFF_LINE_KINDS.find(kind => kind.match(line))?.className
}

// Drop the leading +/-/space gutter character so changes read by color alone
// (like Cursor), keeping the rest of the indentation intact. Hunk headers
// (`@@`) and any stray file headers are left untouched.
function stripDiffMarker(line: string): string {
  if (line.startsWith('@@')) {
    return line
  }

  if ((line.startsWith('+') && !line.startsWith('+++')) || (line.startsWith('-') && !line.startsWith('---'))) {
    return line.slice(1)
  }

  if (line.startsWith(' ')) {
    return line.slice(1)
  }

  return line
}

interface DisplayLine {
  className?: string
  text: string
}

// Build the rendered line list: drop `@@ … @@` hunk headers (git noise in a
// GUI) and the +/- gutter, but keep a blank separator between hunks so
// multi-hunk diffs don't visually merge.
function toDisplayLines(text: string): DisplayLine[] {
  const out: DisplayLine[] = []
  let emitted = false

  for (const line of text.split('\n')) {
    if (line.startsWith('@@')) {
      if (emitted) {
        out.push({ text: '' })
      }

      continue
    }

    out.push({ className: classifyLine(line), text: stripDiffMarker(line) })
    emitted = true
  }

  return out
}

interface DiffLinesProps extends Omit<React.ComponentProps<'pre'>, 'children'> {
  text: string
}

export function DiffLines({ className, text, ...props }: DiffLinesProps) {
  const lines = React.useMemo(() => toDisplayLines(text), [text])

  return (
    <pre
      className={cn(
        'max-h-[12rem] max-w-full min-w-0 overflow-auto overscroll-contain px-0 py-1 font-mono text-[0.7rem] leading-relaxed text-(--ui-text-secondary)',
        className
      )}
      data-slot="diff-lines"
      {...props}
    >
      {lines.map((line, index) => (
        <span
          className={cn('block min-w-max border-l-2 border-transparent whitespace-pre px-2.5 py-px', line.className)}
          key={`${index}-${line.text}`}
        >
          {line.text || ' '}
        </span>
      ))}
    </pre>
  )
}

// Git-style unified diffs arrive with a file-header preamble — `diff --git`,
// `index …`, `--- a/path`, `+++ b/path`, and Hermes' own `a/path → b/path`
// arrow line. That preamble just repeats the path (which the tool row already
// shows) and reads especially badly for absolute paths (`a//Users/…`). Strip
// the leading header zone up to the first hunk so the panel shows only hunks +
// changes, the way Cursor does.
const DIFF_HEADER_PREFIXES = ['diff --git', 'index ', '--- ', '+++ ', 'similarity ', 'rename ', 'new file', 'deleted file']

function isArrowHeaderLine(line: string): boolean {
  const trimmed = line.trim()

  return trimmed.includes('→') && /^\S.*→\s*\S+$/.test(trimmed) && !/^[+\-@]/.test(trimmed)
}

/** Exported for tests. */
export function stripDiffFileHeaders(diff: string): string {
  const lines = diff.split('\n')
  let start = 0

  for (; start < lines.length; start += 1) {
    const line = lines[start]

    if (line.startsWith('@@')) {
      break
    }

    if (line.trim() === '' || isArrowHeaderLine(line) || DIFF_HEADER_PREFIXES.some(prefix => line.startsWith(prefix))) {
      continue
    }

    break
  }

  return lines.slice(start).join('\n')
}

interface FileDiffPanelProps {
  diff: string
}

export function FileDiffPanel({ diff }: FileDiffPanelProps) {
  const display = React.useMemo(() => stripDiffFileHeaders(diff), [diff])

  // Bleed out of the tool-card body's `p-1.5` so changed-line tints/borders run
  // flush to the card edges (rounded corners clip via the card's overflow).
  // `max-w-none` lifts the base `max-w-full` cap that would otherwise stop the
  // negative margins from widening the block.
  return <DiffLines className="-mx-1.5 -mb-1.5 max-w-none" data-slot="file-diff-panel" text={display} />
}

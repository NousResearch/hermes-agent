'use client'

import type { ReactNode } from 'react'
import * as React from 'react'
import { useShikiHighlighter } from 'react-shiki'
import type { ShikiTransformer } from 'shiki'

import { exceedsHighlightBudget, SHIKI_THEME } from '@/components/chat/shiki-highlighter'
import { shikiLanguageForFilename } from '@/lib/markdown-code'
import { cn } from '@/lib/utils'

/**
 * Renders a unified diff for a tool's file edit. Two paths share one parse:
 *  - `SyntaxDiff` highlights the change *content* in the file's language via
 *    Shiki, then a per-line transformer paints the add/remove tint on top.
 *  - `DiffLines` is the color-only fallback (no language, over budget, or while
 *    Shiki loads).
 * Both drop git file-headers + `@@` hunk noise and the `+/-` gutter so changes
 * read by color + a 2px gutter accent, the way Cursor does.
 */
type DiffKind = 'add' | 'context' | 'remove'

interface DiffLine {
  kind: DiffKind
  text: string
}

// Tint + 2px gutter accent per change kind. Text color is included for the
// plain renderer; the Shiki path omits it so syntax colors win, layering only
// the background + border.
const DIFF_KIND_TINT: Record<DiffKind, string> = {
  add: 'border-emerald-500 bg-emerald-500/12',
  context: 'border-transparent',
  remove: 'border-rose-500 bg-rose-500/12'
}

const DIFF_KIND_TEXT: Record<DiffKind, string> = {
  add: 'text-emerald-800 dark:text-emerald-200',
  context: '',
  remove: 'text-rose-800 dark:text-rose-200'
}

const DIFF_LINE_BASE = 'block min-w-max whitespace-pre border-l-2 px-2.5 py-px'

// Bleed out of the tool-card body's `p-1.5` so tints/borders run flush to the
// card edges (rounded corners clip via the card's overflow); compact height
// with internal scroll like a code block.
const DIFF_BOX_CLASS =
  '-mx-1.5 -mb-1.5 max-h-[12rem] max-w-none min-w-0 overflow-auto overscroll-contain font-mono text-[0.7rem] leading-relaxed text-(--ui-text-secondary)'

function diffKind(line: string): DiffKind {
  if (line.startsWith('+') && !line.startsWith('+++')) {
    return 'add'
  }

  if (line.startsWith('-') && !line.startsWith('---')) {
    return 'remove'
  }

  return 'context'
}

// Drop the leading +/-/space gutter so changes read by color alone, keeping the
// rest of the indentation intact.
function stripDiffMarker(line: string): string {
  if (diffKind(line) !== 'context' || line.startsWith(' ')) {
    return line.slice(1)
  }

  return line
}

// Git-style unified diffs arrive with a file-header preamble — `diff --git`,
// `index …`, `--- a/path`, `+++ b/path`, and Hermes' own `a/path → b/path`
// arrow line. That preamble just repeats the path (which the tool row already
// shows) and reads especially badly for absolute paths (`a//Users/…`). Strip
// the leading header zone up to the first hunk.
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

// Cleaned diff → renderable lines: file-headers + `@@` hunks dropped (a blank
// separator kept between hunks), markers stripped, kind recorded.
function parseDiff(diff: string): DiffLine[] {
  const out: DiffLine[] = []
  let emitted = false

  for (const line of stripDiffFileHeaders(diff).split('\n')) {
    if (line.startsWith('@@')) {
      if (emitted) {
        out.push({ kind: 'context', text: '' })
      }

      continue
    }

    out.push({ kind: diffKind(line), text: stripDiffMarker(line) })
    emitted = true
  }

  return out
}

function DiffBody({ lines, syntax }: { lines: DiffLine[]; syntax?: boolean }) {
  return (
    <>
      {lines.map((line, index) => (
        <span
          className={cn(DIFF_LINE_BASE, DIFF_KIND_TINT[line.kind], !syntax && DIFF_KIND_TEXT[line.kind])}
          key={`${index}-${line.text}`}
        >
          {line.text || ' '}
        </span>
      ))}
    </>
  )
}

// Shiki transformer: tag each `.line` with the diff tint for its kind, so the
// syntax-highlighted output keeps add/remove backgrounds + the gutter accent.
function diffLineTransformer(kinds: DiffKind[]): ShikiTransformer {
  return {
    line(node, line) {
      const kind = kinds[line - 1] ?? 'context'

      const existing = Array.isArray(node.properties.className)
        ? (node.properties.className as string[])
        : node.properties.className
          ? [String(node.properties.className)]
          : []

      node.properties.className = [...existing, DIFF_LINE_BASE, DIFF_KIND_TINT[kind]]
    }
  }
}

function SyntaxDiff({ language, lines }: { language: string; lines: DiffLine[] }) {
  const code = React.useMemo(() => lines.map(line => line.text).join('\n'), [lines])
  const transformers = React.useMemo(() => [diffLineTransformer(lines.map(line => line.kind))], [lines])

  const highlighted = useShikiHighlighter(code, language, SHIKI_THEME, {
    defaultColor: 'light-dark()',
    transformers
  })

  // Until Shiki resolves, show the plain colored diff so there's no flash.
  return (highlighted as ReactNode) ?? <DiffBody lines={lines} />
}

interface DiffLinesProps extends Omit<React.ComponentProps<'pre'>, 'children'> {
  text: string
}

export function DiffLines({ className, text, ...props }: DiffLinesProps) {
  const lines = React.useMemo(() => parseDiff(text), [text])

  return (
    <pre className={cn(DIFF_BOX_CLASS, className)} data-slot="diff-lines" {...props}>
      <DiffBody lines={lines} />
    </pre>
  )
}

interface FileDiffPanelProps {
  diff: string
  path?: string
}

export function FileDiffPanel({ diff, path }: FileDiffPanelProps) {
  const lines = React.useMemo(() => parseDiff(diff), [diff])
  const language = shikiLanguageForFilename(path)
  const canHighlight = Boolean(language) && !exceedsHighlightBudget(diff)

  return (
    <div className={DIFF_BOX_CLASS} data-slot="file-diff-panel">
      {canHighlight ? <SyntaxDiff language={language} lines={lines} /> : <DiffBody lines={lines} />}
    </div>
  )
}

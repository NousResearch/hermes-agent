'use client'

import type { ReactNode } from 'react'
import * as React from 'react'
import { useShikiHighlighter } from 'react-shiki'
import { type BundledLanguage, codeToTokens, type ShikiTransformer, type ThemedToken } from 'shiki'

import { chunkLines, type LineChunk, useFixedRowWindow } from '@/components/chat/fixed-row-window'
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

export interface DiffLine {
  kind: DiffKind
  text: string
  /** 1-based line number in the old/new file (absent on the "other" side of an
   *  add/remove, and on hunk-separator blanks). Only used when line numbers are
   *  shown (the preview's full diff). */
  newNo?: number
  oldNo?: number
}

interface ParsedHunk {
  lines: Array<{ kind: DiffKind; text: string }>
  newStart: number
  oldStart: number
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
const PREVIEW_DIFF_LINE_BASE = 'block h-5 min-w-max whitespace-pre px-2.5 leading-5'
const PREVIEW_CHUNK_LINES = 200
const PREVIEW_LINE_PX = 20
const PREVIEW_OVERSCAN_LINES = 400

// Bleed out of the tool-card body's `p-1.5` so tints/borders run flush to the
// card edges (rounded corners clip via the card's overflow); compact height
// with internal scroll like a code block.
// `overscroll-y-auto` so reaching the box's top/bottom hands the wheel back to
// the page (no scroll-trap); `overscroll-x-contain` keeps a trackpad's sideways
// overscroll on long code lines from firing browser back/forward navigation.
const DIFF_BOX_CLASS =
  '-mx-1.5 -mb-1.5 max-h-[12rem] max-w-none min-w-0 overflow-auto overscroll-x-contain overscroll-y-auto font-mono text-[0.7rem] leading-relaxed text-(--ui-text-secondary)'

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
// `index …`, `--- a/path`, `+++ b/path`, and IX Agency' own `a/path → b/path`
// arrow line. That preamble just repeats the path (which the tool row already
// shows) and reads especially badly for absolute paths (`a//Users/…`). Strip
// the leading header zone up to the first hunk.
const DIFF_HEADER_PREFIXES = [
  'diff --git',
  'index ',
  '--- ',
  '+++ ',
  'similarity ',
  'rename ',
  'new file',
  'deleted file'
]

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

function parseHunks(diff: string): ParsedHunk[] {
  const hunks: ParsedHunk[] = []
  let active: null | ParsedHunk = null

  for (const line of stripDiffFileHeaders(diff).split('\n')) {
    if (line.startsWith('@@')) {
      const match = /@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/.exec(line)

      if (!match) {
        active = null

        continue
      }

      active = { oldStart: Number(match[1]), newStart: Number(match[2]), lines: [] }
      hunks.push(active)

      continue
    }

    if (!active || line.startsWith('\\')) {
      continue
    }

    active.lines.push({ kind: diffKind(line), text: stripDiffMarker(line) })
  }

  return hunks
}

// Cleaned diff → renderable lines: file-headers + `@@` hunks dropped (a blank
// separator kept between hunks), markers stripped, kind recorded. Old/new line
// numbers are tracked from each `@@ -a,b +c,d @@` header so a caller that wants
// a gutter (the preview) can render them; the blank separator carries none.
function parseDiff(diff: string): DiffLine[] {
  const hunks = parseHunks(diff)

  if (hunks.length === 0) {
    // Fallback for unexpected non-hunk payloads.
    return stripDiffFileHeaders(diff)
      .split('\n')
      .map(line => ({ kind: diffKind(line), text: stripDiffMarker(line) }))
  }

  const out: DiffLine[] = []
  let emitted = false
  let oldNo = 1
  let newNo = 1

  for (const hunk of hunks) {
    oldNo = hunk.oldStart
    newNo = hunk.newStart

    if (emitted) {
      out.push({ kind: 'context', text: '' })
    }

    for (const line of hunk.lines) {
      const entry: DiffLine = { kind: line.kind, text: line.text }

      if (line.kind === 'add') {
        entry.newNo = newNo++
      } else if (line.kind === 'remove') {
        entry.oldNo = oldNo++
      } else {
        entry.oldNo = oldNo++
        entry.newNo = newNo++
      }

      out.push(entry)
      emitted = true
    }
  }

  return out
}

// Build a full-file diff view anchored to the CURRENT file text. Every current
// line is emitted from `fullText` with its real new-file line number; hunks only
// mark those rows as added and insert deleted rows between them. That keeps the
// preview's SOURCE and DIFF views on the same line map even when git returns
// compact hunks or removed-only rows.
function parseFullFileDiff(diff: string, fullText: string): DiffLine[] {
  const hunks = parseHunks(diff)
  const fullLines = fullText.split('\n')

  if (hunks.length === 0) {
    return fullLines.map((text, index) => ({ kind: 'context', newNo: index + 1, oldNo: index + 1, text }))
  }

  const added = new Set<number>()
  const oldNoByNewNo = new Map<number, number>()
  const removalsByNewNo = new Map<number, DiffLine[]>()
  const out: DiffLine[] = []

  for (const hunk of hunks) {
    let oldNo = hunk.oldStart
    let newNo = hunk.newStart

    for (const line of hunk.lines) {
      if (line.kind === 'add') {
        added.add(newNo)
        newNo += 1
      } else if (line.kind === 'remove') {
        const anchor = Math.max(1, Math.min(newNo, fullLines.length + 1))
        const bucket = removalsByNewNo.get(anchor) ?? []

        bucket.push({ kind: 'remove', oldNo, text: line.text })
        removalsByNewNo.set(anchor, bucket)
        oldNo += 1
      } else {
        oldNoByNewNo.set(newNo, oldNo)
        oldNo += 1
        newNo += 1
      }
    }
  }

  for (let index = 0; index < fullLines.length; index += 1) {
    const newNo = index + 1
    const removals = removalsByNewNo.get(newNo)

    if (removals) {
      out.push(...removals)
    }

    out.push({
      kind: added.has(newNo) ? 'add' : 'context',
      newNo,
      oldNo: oldNoByNewNo.get(newNo),
      text: fullLines[index] ?? ''
    })
  }

  const trailingRemovals = removalsByNewNo.get(fullLines.length + 1)

  if (trailingRemovals) {
    out.push(...trailingRemovals)
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

// shiki FontStyle is a bitmask: Italic=1, Bold=2, Underline=4.
function tokenStyle({ bgColor, color, fontStyle = 0 }: ThemedToken): React.CSSProperties | undefined {
  if (!color && !bgColor && !fontStyle) {
    return undefined
  }

  return {
    backgroundColor: bgColor,
    color,
    fontStyle: fontStyle & 1 ? 'italic' : undefined,
    fontWeight: fontStyle & 2 ? 700 : undefined,
    textDecorationLine: fontStyle & 4 ? 'underline' : undefined
  }
}

function useThemeName() {
  const current = () => (document.documentElement.classList.contains('dark') ? SHIKI_THEME.dark : SHIKI_THEME.light)
  const [theme, setTheme] = React.useState(current)

  React.useEffect(() => {
    const observer = new MutationObserver(() => setTheme(current()))

    observer.observe(document.documentElement, { attributeFilter: ['class'], attributes: true })

    return () => observer.disconnect()
  }, [])

  return theme
}

function PreviewDiffRows({
  afterLines = 0,
  beforeLines = 0,
  chunks,
  tokens
}: {
  afterLines?: number
  beforeLines?: number
  chunks: Array<LineChunk<DiffLine>>
  tokens?: ThemedToken[][] | null
}) {
  return (
    <>
      {beforeLines > 0 && <div aria-hidden style={{ height: beforeLines * PREVIEW_LINE_PX }} />}
      {chunks.map(chunk => (
        <div className="block" key={chunk.start}>
          {chunk.lines.map((line, offset) => {
            const index = chunk.start + offset
            const rowTokens = tokens?.[index] ?? []

            return (
              <span className={cn(PREVIEW_DIFF_LINE_BASE, DIFF_KIND_TINT[line.kind])} key={`${index}-${line.text}`}>
                {rowTokens.length > 0
                  ? rowTokens.map((token, tokenIndex) => (
                      <span key={`${tokenIndex}-${token.offset}`} style={tokenStyle(token)}>
                        {token.content}
                      </span>
                    ))
                  : line.text || ' '}
              </span>
            )
          })}
        </div>
      ))}
      {afterLines > 0 && <div aria-hidden style={{ height: afterLines * PREVIEW_LINE_PX }} />}
    </>
  )
}

function TokenizedDiffBody({
  afterLines,
  beforeLines,
  chunked = false,
  chunks,
  language,
  lines
}: {
  afterLines?: number
  beforeLines?: number
  chunked?: boolean
  chunks?: Array<LineChunk<DiffLine>>
  language: string
  lines: DiffLine[]
}) {
  const code = React.useMemo(() => lines.map(line => line.text).join('\n'), [lines])
  const theme = useThemeName()
  const [tokens, setTokens] = React.useState<ThemedToken[][] | null>(null)

  React.useEffect(() => {
    let cancelled = false

    setTokens(null)
    void codeToTokens(code, { lang: language as BundledLanguage, theme })
      .then(result => {
        if (!cancelled) {
          setTokens(result.tokens)
        }
      })
      .catch(() => {
        if (!cancelled) {
          setTokens([])
        }
      })

    return () => {
      cancelled = true
    }
  }, [code, language, theme])

  if (!tokens) {
    return chunked ? (
      <PreviewDiffRows
        afterLines={afterLines}
        beforeLines={beforeLines}
        chunks={chunks ?? chunkLines(lines, PREVIEW_CHUNK_LINES)}
      />
    ) : (
      <DiffBody lines={lines} />
    )
  }

  if (chunked) {
    return (
      <PreviewDiffRows
        afterLines={afterLines}
        beforeLines={beforeLines}
        chunks={chunks ?? chunkLines(lines, PREVIEW_CHUNK_LINES)}
        tokens={tokens}
      />
    )
  }

  return (
    <>
      {lines.map((line, index) => {
        const rowTokens = tokens[index] ?? []

        return (
          <span className={cn(PREVIEW_DIFF_LINE_BASE, DIFF_KIND_TINT[line.kind])} key={`${index}-${line.text}`}>
            {rowTokens.length > 0
              ? rowTokens.map((token, tokenIndex) => (
                  <span key={`${tokenIndex}-${token.offset}`} style={tokenStyle(token)}>
                    {token.content}
                  </span>
                ))
              : line.text || ' '}
          </span>
        )
      })}
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

// Coalesce consecutive same-kind changed rows into runs, each placed by line
// fraction (no DOM measurement). Context rows produce no tick.
function overviewRuns(lines: DiffLine[]): { kind: 'add' | 'remove'; sizePct: number; startPct: number }[] {
  const total = lines.length || 1
  const runs: { kind: 'add' | 'remove'; sizePct: number; startPct: number }[] = []

  for (let i = 0; i < lines.length; ) {
    const kind = lines[i].kind

    if (kind === 'context') {
      i += 1

      continue
    }

    let j = i + 1

    while (j < lines.length && lines[j].kind === kind) {
      j += 1
    }

    runs.push({ kind, sizePct: ((j - i) / total) * 100, startPct: (i / total) * 100 })
    i = j
  }

  return runs
}

// VS Code-style overview ruler: a thin strip pinned to the diff's right edge with
// a green/red tick per change, positioned by line fraction. Pinned to the
// viewport (not the scrolled content) by living as an absolute sibling of the
// scroller inside a relative wrapper — so no scroll listener or measurement.
function DiffOverviewRuler({ lines }: { lines: DiffLine[] }) {
  const runs = React.useMemo(() => overviewRuns(lines), [lines])

  if (runs.length === 0) {
    return null
  }

  return (
    <div aria-hidden className="pointer-events-none absolute top-0 right-0 bottom-0 w-1.5 opacity-80">
      {/* Cap the tick field to the diff's natural height (rows × line px) so a
          short diff renders thin, line-aligned ticks instead of stretching a few
          changes into gross full-height blocks. A long diff hits the 100% cap and
          compresses into a true overview. */}
      <div className="relative w-full" style={{ height: `min(100%, ${lines.length * PREVIEW_LINE_PX}px)` }}>
        {runs.map((run, index) => (
          <div
            className={cn('absolute inset-x-0', run.kind === 'add' ? 'bg-(--ui-green)' : 'bg-(--ui-red)')}
            key={index}
            style={{ height: `max(0.125rem, ${run.sizePct}%)`, top: `${run.startPct}%` }}
          />
        ))}
      </div>
    </div>
  )
}

interface FileDiffPanelProps {
  /** Override the default (tool-card) box styling — the full-height preview
   *  cancels the bleed/clamp so the diff fills its pane. */
  className?: string
  diff: string
  /** Current file text. When provided, the panel expands hunked diffs into a
   *  full-file view so unchanged lines are preserved between hunks. */
  fullText?: string
  path?: string
  /** Render an old/new line-number gutter (the full preview diff). The compact
   *  tool-card + inline review diff leave this off. */
  showLineNumbers?: boolean
}

export function FileDiffPanel({ className, diff, fullText, path, showLineNumbers = false }: FileDiffPanelProps) {
  const lines = React.useMemo(
    () => (fullText != null ? parseFullFileDiff(diff, fullText) : parseDiff(diff)),
    [diff, fullText]
  )

  const lineChunks = React.useMemo(() => chunkLines(lines, PREVIEW_CHUNK_LINES), [lines])

  const { afterRows, beforeRows, endChunk, onScroll, scrollerRef, startChunk } = useFixedRowWindow({
    overscanRows: PREVIEW_OVERSCAN_LINES,
    rowPx: PREVIEW_LINE_PX,
    rowsPerChunk: PREVIEW_CHUNK_LINES,
    totalRows: lines.length
  })

  const visibleLineChunks = lineChunks.slice(startChunk, endChunk + 1)

  const language = shikiLanguageForFilename(path)
  const canHighlight = Boolean(language) && !exceedsHighlightBudget(fullText ?? diff)

  // Full-file preview: we own the rows (tokens rendered inside) so blank lines
  // can't collapse. Compact tool/review diffs let Shiki own the rows.
  const body = !canHighlight ? (
    showLineNumbers ? (
      <PreviewDiffRows afterLines={afterRows} beforeLines={beforeRows} chunks={visibleLineChunks} />
    ) : (
      <DiffBody lines={lines} />
    )
  ) : fullText != null ? (
    <TokenizedDiffBody
      afterLines={afterRows}
      beforeLines={beforeRows}
      chunked={showLineNumbers}
      chunks={visibleLineChunks}
      language={language}
      lines={lines}
    />
  ) : (
    <SyntaxDiff language={language} lines={lines} />
  )

  if (!showLineNumbers) {
    return (
      <div className={cn(DIFF_BOX_CLASS, className)} data-slot="file-diff-panel">
        {body}
      </div>
    )
  }

  // A single line-number gutter (VS Code's inline-diff style): each row shows its
  // own file's number — the new number for context/adds, the old number for
  // removals — with an overview ruler pinned to the right edge. The inner div
  // owns the scroll so the ruler (an absolute sibling) stays viewport-fixed.
  return (
    <div className={cn(DIFF_BOX_CLASS, 'relative overflow-hidden', className)} data-slot="file-diff-panel">
      <div className="absolute inset-0 overflow-auto pr-2.5" onScroll={onScroll} ref={scrollerRef}>
        <div className="grid min-w-max grid-cols-[auto_minmax(0,1fr)]">
          <div className="sticky left-0 z-1 select-none bg-(--ui-editor-surface-background) py-3 text-muted-foreground/55">
            {beforeRows > 0 && <div aria-hidden style={{ height: beforeRows * PREVIEW_LINE_PX }} />}
            {visibleLineChunks.map(chunk => (
              <div className="block" key={chunk.start}>
                {chunk.lines.map((line, offset) => {
                  const index = chunk.start + offset

                  return (
                    <div
                      className="h-5 w-9 pr-2 text-right leading-5 tabular-nums"
                      key={`${index}-${line.oldNo}-${line.newNo}`}
                    >
                      {line.newNo ?? ''}
                    </div>
                  )
                })}
              </div>
            ))}
            {afterRows > 0 && <div aria-hidden style={{ height: afterRows * PREVIEW_LINE_PX }} />}
          </div>
          <div className="min-w-0">{body}</div>
        </div>
      </div>
      <DiffOverviewRuler lines={lines} />
    </div>
  )
}

import type * as React from 'react'
import type {
  ComponentProps,
  CSSProperties,
  DragEvent as ReactDragEvent,
  MouseEvent as ReactMouseEvent,
  ReactNode
} from 'react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ShikiHighlighter from 'react-shiki'
import { Streamdown } from 'streamdown'

import { requestComposerFocus, requestComposerInsertRefs } from '@/app/chat/composer/focus'
import { droppedFileInlineRef } from '@/app/chat/composer/inline-refs'
import { HERMES_PATHS_MIME } from '@/app/chat/hooks/use-composer-actions'
import { isAddSelectionShortcut } from '@/app/right-sidebar/terminal/selection'
import { PageLoader } from '@/components/page-loader'
import { translateNow, useI18n } from '@/i18n'
import { readDesktopFileDataUrl, readDesktopFileText } from '@/lib/desktop-fs'
import { cn } from '@/lib/utils'
import type { PreviewTarget } from '@/store/preview'
import { $currentCwd } from '@/store/session'

const SHIKI_THEME = { dark: 'github-dark-default', light: 'github-light-default' } as const
const TEXT_PREVIEW_MAX_BYTES = 512 * 1024

type EmptyStateTone = 'neutral' | 'warning'

const TONE_STYLES: Record<EmptyStateTone, { cube: string; primary: string }> = {
  neutral: {
    cube: 'text-muted-foreground/35',
    primary: 'border-border bg-background text-foreground hover:bg-accent'
  },
  warning: {
    cube: 'text-amber-500/70 dark:text-amber-300/70',
    primary:
      'border-amber-400/40 bg-amber-50 text-amber-900 hover:bg-amber-100 dark:border-amber-300/30 dark:bg-amber-300/15 dark:text-amber-100 dark:hover:bg-amber-300/20'
  }
}

function PreviewCubeIcon({ className }: { className?: string }) {
  return (
    <svg aria-hidden="true" className={cn('size-16', className)} viewBox="0 0 64 64">
      <path
        d="M32 5 56 18.5v27L32 59 8 45.5v-27L32 5Z"
        fill="none"
        stroke="currentColor"
        strokeLinejoin="round"
        strokeWidth="1.25"
      />
      <path
        d="M8 18.5 32 32l24-13.5M32 32v27"
        fill="none"
        stroke="currentColor"
        strokeLinejoin="round"
        strokeWidth="1.25"
      />
      <path d="M20 11.75 44 25.25" fill="none" opacity="0.45" stroke="currentColor" strokeWidth="0.9" />
    </svg>
  )
}

interface PreviewEmptyStateProps {
  body?: ReactNode
  consoleHeight?: number
  primaryAction?: { disabled?: boolean; label: string; onClick: () => void }
  secondaryAction?: { disabled?: boolean; label: string; onClick: () => void }
  title: string
  tone?: EmptyStateTone
}

export function PreviewEmptyState({
  body,
  consoleHeight = 0,
  primaryAction,
  secondaryAction,
  title,
  tone = 'neutral'
}: PreviewEmptyStateProps) {
  const styles = TONE_STYLES[tone]

  return (
    <div
      className="absolute inset-x-0 top-0 z-10 grid place-items-center bg-background px-8 py-10 text-center bottom-(--preview-error-bottom)"
      style={{ '--preview-error-bottom': `${consoleHeight}px` } as CSSProperties}
    >
      <div className="grid max-w-sm justify-items-center gap-5">
        <PreviewCubeIcon className={styles.cube} />
        <div className="grid gap-2">
          <div className="text-sm font-medium text-foreground">{title}</div>
          {body && <div className="text-xs leading-relaxed text-muted-foreground">{body}</div>}
        </div>
        {(primaryAction || secondaryAction) && (
          <div className="grid justify-items-center gap-2">
            {primaryAction && (
              <button
                className={cn(
                  'rounded-full border px-3.5 py-1.5 text-xs font-medium shadow-xs transition-colors disabled:cursor-default disabled:opacity-60',
                  styles.primary
                )}
                disabled={primaryAction.disabled}
                onClick={primaryAction.onClick}
                type="button"
              >
                {primaryAction.label}
              </button>
            )}
            {secondaryAction && (
              <button
                className="text-[0.6875rem] font-medium text-muted-foreground underline decoration-current/20 underline-offset-4 transition-colors hover:text-foreground disabled:cursor-default disabled:text-muted-foreground/55 disabled:no-underline"
                disabled={secondaryAction.disabled}
                onClick={secondaryAction.onClick}
                type="button"
              >
                {secondaryAction.label}
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

interface LocalPreviewState {
  binary?: boolean
  byteSize?: number
  dataUrl?: string
  error?: string
  language?: string
  loading: boolean
  text?: string
  truncated?: boolean
}

function filePathForTarget(target: PreviewTarget) {
  if (target.path) {
    return target.path
  }

  try {
    const url = new URL(target.url)

    return url.protocol === 'file:' ? decodeURIComponent(url.pathname) : target.url
  } catch {
    return target.url
  }
}

function formatBytes(bytes: number | undefined) {
  if (!bytes) {
    return translateNow('preview.unknownSize')
  }

  const units = ['B', 'KB', 'MB', 'GB']
  let value = bytes
  let unit = 0

  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024
    unit += 1
  }

  return `${value >= 10 || unit === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[unit]}`
}

function looksBinaryBytes(bytes: Uint8Array) {
  if (!bytes.length) {
    return false
  }

  let suspicious = 0

  for (const byte of bytes.slice(0, 4096)) {
    if (byte === 0) {
      return true
    }

    if (byte < 32 && byte !== 9 && byte !== 10 && byte !== 13) {
      suspicious += 1
    }
  }

  return suspicious / Math.min(bytes.length, 4096) > 0.12
}

async function readTextPreview(filePath: string) {
  try {
    return await readDesktopFileText(filePath)
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)

    if (!message.includes("No handler registered for 'hermes:readFileText'")) {
      throw error
    }
  }

  // Back-compat for a running Electron process whose preload hasn't been
  // restarted since readFileText was added. readFileDataUrl already existed.
  const dataUrl = await window.hermesDesktop.readFileDataUrl(filePath)
  const [, metadata = '', data = ''] = dataUrl.match(/^data:([^,]*),(.*)$/) || []
  const base64 = metadata.includes(';base64')
  const mimeType = metadata.replace(/;base64$/, '') || undefined
  const raw = base64 ? atob(data) : decodeURIComponent(data)
  const bytes = Uint8Array.from(raw, ch => ch.charCodeAt(0))

  return {
    binary: looksBinaryBytes(bytes),
    byteSize: bytes.byteLength,
    mimeType,
    path: filePath,
    text: new TextDecoder().decode(bytes)
  }
}

// Lightweight markdown renderer for file previews. Streamdown does the parse;
// our components keep typography simple and route fenced code through Shiki
// without the library's copy/download/fullscreen chrome.
const MD_TAG_CLASSES = {
  h1: 'mb-3 mt-6 text-3xl font-bold leading-tight tracking-tight first:mt-0',
  h2: 'mb-2.5 mt-5 text-2xl font-semibold leading-snug tracking-tight first:mt-0',
  h3: 'mb-2 mt-4 text-xl font-semibold leading-snug first:mt-0',
  h4: 'mb-2 mt-3 text-base font-semibold leading-snug first:mt-0',
  p: 'mb-4 leading-relaxed text-foreground last:mb-0',
  ul: 'mb-4 list-disc pl-6 marker:text-muted-foreground/70 last:mb-0',
  ol: 'mb-4 list-decimal pl-6 marker:text-muted-foreground/70 last:mb-0',
  li: 'mt-1 leading-relaxed',
  blockquote: 'mb-4 border-l-2 border-border pl-3 text-muted-foreground italic last:mb-0',
  pre: 'mb-4 overflow-hidden rounded-lg border border-border bg-card font-mono text-xs leading-relaxed last:mb-0 [&_pre]:m-0 [&_pre]:overflow-x-auto [&_pre]:bg-transparent! [&_pre]:p-3 [&_pre]:font-mono'
} as const

function tagged<T extends keyof typeof MD_TAG_CLASSES>(Tag: T) {
  const base = MD_TAG_CLASSES[Tag]

  const Component = (({ className, ...rest }: ComponentProps<T>) => {
    const Element = Tag as React.ElementType

    return <Element className={cn(base, className)} {...rest} />
  }) as React.FC<ComponentProps<T>>

  Component.displayName = `Md.${Tag}`

  return Component
}

function MarkdownCode({ className, children, ...props }: ComponentProps<'code'>) {
  const language = /language-([^\s]+)/.exec(className || '')?.[1]

  if (!language) {
    return (
      <code
        className={cn(
          'rounded bg-muted px-1 py-0.5 font-mono text-[0.86em] text-pink-700 dark:text-pink-300',
          className
        )}
        {...props}
      >
        {children}
      </code>
    )
  }

  return (
    <ShikiHighlighter
      addDefaultStyles={false}
      as="div"
      defaultColor="light-dark()"
      delay={80}
      language={language}
      showLanguage={false}
      theme={SHIKI_THEME}
    >
      {String(children).replace(/\n$/, '')}
    </ShikiHighlighter>
  )
}

const MARKDOWN_COMPONENTS = {
  h1: tagged('h1'),
  h2: tagged('h2'),
  h3: tagged('h3'),
  h4: tagged('h4'),
  p: tagged('p'),
  ul: tagged('ul'),
  ol: tagged('ol'),
  li: tagged('li'),
  blockquote: tagged('blockquote'),
  pre: tagged('pre'),
  code: MarkdownCode
}

function MarkdownPreview({ text }: { text: string }) {
  return (
    <div className="preview-markdown mx-auto max-w-3xl px-4 py-3 text-sm text-foreground" data-selectable-text="true">
      <Streamdown components={MARKDOWN_COMPONENTS} controls={false} mode="static" parseIncompleteMarkdown={false}>
        {text}
      </Streamdown>
    </div>
  )
}

// --- DiffView: full-file view with git changes highlighted ---

type DiffLineType = 'add' | 'context' | 'remove'

interface FullDiffLine {
  content: string
  lineNum: number
  segments?: DiffSegment[]
  type: DiffLineType
}

export interface DiffSegment {
  changed: boolean
  text: string
}

const DIFF_TOKEN_RE = /(\s+|[\p{L}\p{N}_]+|[^\s\p{L}\p{N}_])/gu

function tokenizeDiffText(text: string): string[] {
  return text.match(DIFF_TOKEN_RE) ?? []
}

function mergeDiffSegments(tokens: string[], unchanged: Set<number>): DiffSegment[] {
  const segments: DiffSegment[] = []

  tokens.forEach((text, index) => {
    const changed = !unchanged.has(index)
    const previous = segments.at(-1)

    if (previous?.changed === changed) {
      previous.text += text
    } else {
      segments.push({ changed, text })
    }
  })

  return segments
}

export function computeTokenDiff(
  previousText: string,
  nextText: string
): { next: DiffSegment[]; previous: DiffSegment[] } {
  const previousTokens = tokenizeDiffText(previousText)
  const nextTokens = tokenizeDiffText(nextText)
  const rows = previousTokens.length + 1
  const columns = nextTokens.length + 1
  const lcs = Array.from({ length: rows }, () => new Uint32Array(columns))

  for (let previousIndex = 1; previousIndex < rows; previousIndex += 1) {
    for (let nextIndex = 1; nextIndex < columns; nextIndex += 1) {
      lcs[previousIndex][nextIndex] =
        previousTokens[previousIndex - 1] === nextTokens[nextIndex - 1]
          ? lcs[previousIndex - 1][nextIndex - 1] + 1
          : Math.max(lcs[previousIndex - 1][nextIndex], lcs[previousIndex][nextIndex - 1])
    }
  }

  const unchangedPrevious = new Set<number>()
  const unchangedNext = new Set<number>()
  let previousIndex = previousTokens.length
  let nextIndex = nextTokens.length

  while (previousIndex > 0 && nextIndex > 0) {
    if (previousTokens[previousIndex - 1] === nextTokens[nextIndex - 1]) {
      unchangedPrevious.add(previousIndex - 1)
      unchangedNext.add(nextIndex - 1)
      previousIndex -= 1
      nextIndex -= 1
    } else if (lcs[previousIndex - 1][nextIndex] >= lcs[previousIndex][nextIndex - 1]) {
      previousIndex -= 1
    } else {
      nextIndex -= 1
    }
  }

  return {
    next: mergeDiffSegments(nextTokens, unchangedNext),
    previous: mergeDiffSegments(previousTokens, unchangedPrevious)
  }
}

function addTokenEmphasis(lines: FullDiffLine[]): FullDiffLine[] {
  const emphasized = lines.map(line => ({ ...line }))

  for (let index = 0; index < emphasized.length; ) {
    if (emphasized[index].type !== 'remove') {
      index += 1

      continue
    }

    const removedStart = index

    while (index < emphasized.length && emphasized[index].type === 'remove') {
      index += 1
    }

    const addedStart = index

    while (index < emphasized.length && emphasized[index].type === 'add') {
      index += 1
    }

    const pairCount = Math.min(addedStart - removedStart, index - addedStart)

    for (let pairIndex = 0; pairIndex < pairCount; pairIndex += 1) {
      const removedLine = emphasized[removedStart + pairIndex]
      const addedLine = emphasized[addedStart + pairIndex]
      const segments = computeTokenDiff(removedLine.content, addedLine.content)

      removedLine.segments = segments.previous
      addedLine.segments = segments.next
    }
  }

  return emphasized
}

/**
 * Compute line-level diff between old (HEAD) and new (working) text.
 * Uses LCS for the middle section after trimming common prefix/suffix.
 */
function computeLineDiff(
  headLines: string[],
  fileLines: string[]
): {
  added: Set<number>
  removed: Array<{ afterLine: number; content: string }>
} {
  const added = new Set<number>()
  const removed: Array<{ afterLine: number; content: string }> = []

  // Find common prefix
  let start = 0

  while (start < headLines.length && start < fileLines.length && headLines[start] === fileLines[start]) {
    start++
  }

  // Find common suffix
  let headEnd = headLines.length
  let fileEnd = fileLines.length

  while (headEnd > start && fileEnd > start && headLines[headEnd - 1] === fileLines[fileEnd - 1]) {
    headEnd--
    fileEnd--
  }

  // Middle sections
  const headMid = headLines.slice(start, headEnd)
  const fileMid = fileLines.slice(start, fileEnd)

  if (headMid.length === 0 && fileMid.length === 0) {
    return { added, removed }
  }

  // Build LCS table
  const m = headMid.length
  const n = fileMid.length
  const dp: number[][] = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0))

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (headMid[i - 1] === fileMid[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1] + 1
      } else {
        dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1])
      }
    }
  }

  // Backtrack to find LCS
  const lcsIndices = new Array<{ headIdx: number; fileIdx: number }>()

  let i = m,
    j = n

  while (i > 0 && j > 0) {
    if (headMid[i - 1] === fileMid[j - 1]) {
      lcsIndices.push({ headIdx: i - 1, fileIdx: j - 1 })
      i--
      j--
    } else if (dp[i - 1][j] > dp[i][j - 1]) {
      i--
    } else {
      j--
    }
  }

  lcsIndices.reverse()

  // Build annotations
  let hi = 0,
    fi = 0

  for (const { headIdx, fileIdx } of lcsIndices) {
    // Lines in headMid not in LCS → removed
    while (hi < headIdx) {
      const afterLine = start + fi // insert before this file line
      removed.push({ afterLine: afterLine, content: headMid[hi] })
      hi++
    }

    // Lines in fileMid not in LCS → added
    while (fi < fileIdx) {
      added.add(start + fi + 1) // 1-indexed
      fi++
    }

    // Match → context
    hi++
    fi++
  }

  // Trailing removes
  while (hi < headMid.length) {
    removed.push({ afterLine: start + fi, content: headMid[hi] })
    hi++
  }

  // Trailing adds
  while (fi < fileMid.length) {
    added.add(start + fi + 1)
    fi++
  }

  return { added, removed }
}

function DiffView({ fileContent, headContent }: { fileContent: string; headContent: string }) {
  const lines = useMemo(() => {
    const fileLines = fileContent.split('\n')
    const headLines = headContent.split('\n')
    const result: FullDiffLine[] = []

    // Untracked: all lines are added
    if (!headContent && fileContent) {
      for (let i = 0; i < fileLines.length; i++) {
        result.push({ content: fileLines[i], lineNum: i + 1, type: 'add' })
      }

      return result
    }

    const { added, removed } = computeLineDiff(headLines, fileLines)

    // Group removed lines by insertion point
    const removesByLine = new Map<number, string[]>()

    for (const r of removed) {
      const existing = removesByLine.get(r.afterLine) || []
      existing.push(r.content)
      removesByLine.set(r.afterLine, existing)
    }

    for (let i = 0; i < fileLines.length; i++) {
      const lineNum = i + 1

      // Insert removed lines BEFORE this line
      const removes = removesByLine.get(lineNum - 1) || []

      for (const rc of removes) {
        result.push({ content: rc, lineNum: 0, type: 'remove' })
      }

      result.push({
        content: fileLines[i],
        lineNum,
        type: added.has(lineNum) ? 'add' : 'context'
      })
    }

    // Trailing removes
    const trailing = removesByLine.get(fileLines.length) || []

    for (const rc of trailing) {
      result.push({ content: rc, lineNum: 0, type: 'remove' })
    }

    return addTokenEmphasis(result)
  }, [fileContent, headContent])

  const hasChanges = lines.some(l => l.type !== 'context')

  if (!hasChanges) {
    return <div className="grid h-32 place-items-center text-xs text-muted-foreground/60">No uncommitted changes</div>
  }

  return (
    <div className="w-full font-mono text-xs leading-relaxed select-text overflow-x-auto" data-selectable-text="true">
      {lines.map((line, index) => (
        <div
          className={cn(
            'flex',
            line.type === 'add' && 'bg-emerald-500/15 dark:bg-emerald-500/10',
            line.type === 'remove' && 'bg-rose-500/15 dark:bg-rose-500/10'
          )}
          key={index}
        >
          <div
            className={cn(
              'w-12 shrink-0 select-none px-1.5 py-px text-right tabular-nums',
              line.type === 'add' && 'text-emerald-600/60 dark:text-emerald-400/50',
              line.type === 'remove' && 'text-rose-600/60 dark:text-rose-400/50',
              line.type === 'context' && 'text-muted-foreground/40'
            )}
          >
            {line.lineNum || ''}
          </div>
          <div
            className={cn(
              'w-5 shrink-0 select-none py-px text-center',
              line.type === 'add' && 'text-emerald-700 dark:text-emerald-300',
              line.type === 'remove' && 'text-rose-700 dark:text-rose-300'
            )}
          >
            {line.type === 'add' ? '+' : line.type === 'remove' ? '-' : ' '}
          </div>
          <pre
            className={cn(
              'flex-1 whitespace-pre-wrap break-words py-px pr-3',
              line.type === 'add' && 'text-emerald-800 dark:text-emerald-200',
              line.type === 'remove' && 'text-rose-800 dark:text-rose-200',
              line.type === 'context' && 'text-foreground'
            )}
          >
            {line.segments
              ? line.segments.map((segment, segmentIndex) => (
                  <span
                    className={cn(
                      segment.changed &&
                        line.type === 'add' &&
                        'rounded-[0.125rem] bg-emerald-500/35 dark:bg-emerald-400/25',
                      segment.changed &&
                        line.type === 'remove' &&
                        'rounded-[0.125rem] bg-rose-500/35 dark:bg-rose-400/25'
                    )}
                    key={segmentIndex}
                  >
                    {segment.text}
                  </span>
                ))
              : line.content}
          </pre>
        </div>
      ))}
    </div>
  )
}

// --- PreviewToggle: segmented control for SOURCE | CHANGES | PREVIEW ---

type RenderMode = 'diff' | 'preview' | 'source'

function PreviewToggle({
  hasDiff,
  hasPreview,
  mode,
  onModeChange
}: {
  hasDiff: boolean
  hasPreview: boolean
  mode: RenderMode
  onModeChange: (mode: RenderMode) => void
}) {
  const { t } = useI18n()

  const modes: { key: RenderMode; label: string }[] = [
    ...(hasPreview ? [{ key: 'preview' as RenderMode, label: t.preview.renderedPreview }] : []),
    { key: 'source' as RenderMode, label: t.preview.source },
    { key: 'diff' as RenderMode, label: t.preview.changes }
  ]

  return (
    <div className="sticky top-0 z-10 flex justify-end border-b border-border/40 bg-transparent px-3 py-1 backdrop-blur">
      <div className="flex gap-0.5 rounded-md bg-muted/50 p-0.5">
        {modes.map(({ key, label }) => (
          <button
            className={cn(
              'rounded px-2 py-0.5 text-[0.625rem] font-bold transition-colors',
              mode === key ? 'bg-background text-foreground shadow-xs' : 'text-muted-foreground hover:text-foreground'
            )}
            key={key}
            onClick={() => onModeChange(key)}
            type="button"
          >
            {label}
          </button>
        ))}
      </div>
    </div>
  )
}

// Gutter and Shiki output share `font-mono text-xs leading-relaxed py-3` so
// each line aligns vertically. The selection overlay relies on the same
// `text-xs * leading-relaxed = 1.21875rem` line-height to position itself.
const SOURCE_LINE_HEIGHT_REM = 1.21875
const SOURCE_PAD_Y_REM = 0.75

interface LineSelection {
  end: number
  start: number
}

function startLineDrag(event: ReactDragEvent<HTMLElement>, filePath: string, { end, start }: LineSelection) {
  const lineEnd = end > start ? end : undefined
  const label = lineEnd ? `${filePath}:${start}-${end}` : `${filePath}:${start}`

  event.dataTransfer.setData(HERMES_PATHS_MIME, JSON.stringify([{ line: start, lineEnd, path: filePath }]))
  event.dataTransfer.setData('text/plain', label)
  event.dataTransfer.effectAllowed = 'copy'
}

function HighlightedSource({
  language,
  onReady,
  text
}: {
  language: string
  onReady: (text: string) => void
  text: string
}) {
  const hostRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    const host = hostRef.current

    if (!host) {
      return
    }

    let frame = 0

    const reportReady = () => {
      if (!host.querySelector('.shiki')) {
        return
      }

      frame = window.requestAnimationFrame(() => onReady(text))
    }

    const observer = new MutationObserver(reportReady)

    observer.observe(host, { childList: true, subtree: true })
    reportReady()

    return () => {
      observer.disconnect()

      if (frame) {
        window.cancelAnimationFrame(frame)
      }
    }
  }, [onReady, text])

  return (
    <div ref={hostRef}>
      <ShikiHighlighter
        addDefaultStyles={false}
        as="div"
        defaultColor="light-dark()"
        delay={0}
        language={language || 'text'}
        showLanguage={false}
        theme={SHIKI_THEME}
      >
        {text}
      </ShikiHighlighter>
    </div>
  )
}

type EditorStatus = 'editing' | 'saving' | 'view'

export function SourceView({
  filePath,
  language,
  onContentSaved,
  text
}: {
  filePath: string
  language: string
  onContentSaved?: (newContent: string) => void
  text: string
}) {
  const { t } = useI18n()
  const lineCount = useMemo(() => Math.max(1, text.split('\n').length), [text])
  const [selection, setSelection] = useState<LineSelection | null>(null)
  const [status, setStatus] = useState<EditorStatus>('view')
  const [editContent, setEditContent] = useState(text)
  const [pendingSavedText, setPendingSavedText] = useState<string | null>(null)
  const editorRef = useRef<HTMLTextAreaElement | null>(null)
  const filePathRef = useRef(filePath)
  const inSelection = (line: number) => selection != null && line >= selection.start && line <= selection.end
  const editorVisible = status !== 'view'

  useEffect(() => {
    if (filePathRef.current === filePath) {
      return
    }

    filePathRef.current = filePath
    setEditContent(text)
    setPendingSavedText(null)
    setStatus('view')
  }, [filePath, text])

  useEffect(() => {
    if (status === 'view') {
      setEditContent(text)
    }
  }, [status, text])

  useEffect(() => {
    if (status !== 'editing') {
      return
    }

    const frame = window.requestAnimationFrame(() => {
      editorRef.current?.focus()
    })

    return () => window.cancelAnimationFrame(frame)
  }, [status])

  const handleHighlightedReady = useCallback(
    (readyText: string) => {
      if (pendingSavedText !== readyText) {
        return
      }

      setPendingSavedText(null)
      setStatus('view')
    },
    [pendingSavedText]
  )

  const handleSave = async () => {
    const writeFileText = window.hermesDesktop?.writeFileText

    if (!writeFileText || status === 'saving') {
      return
    }

    if (editContent === text) {
      setStatus('view')

      return
    }

    setStatus('saving')

    try {
      const result = await writeFileText(filePath, editContent)

      if (!result?.success) {
        setStatus('editing')

        return
      }

      setPendingSavedText(editContent)
      onContentSaved?.(editContent)
    } catch {
      setStatus('editing')
    }
  }

  const handleLineClick = (event: ReactMouseEvent, line: number) => {
    if (event.shiftKey && selection) {
      setSelection({ end: Math.max(selection.end, line), start: Math.min(selection.start, line) })

      return
    }

    if (selection?.start === line && selection.end === line) {
      setSelection(null)

      return
    }

    setSelection({ end: line, start: line })
  }

  const handleDragStart = (event: ReactDragEvent<HTMLElement>, line: number) => {
    startLineDrag(event, filePath, inSelection(line) && selection ? selection : { end: line, start: line })
  }

  // ⌘/Ctrl+L with a line selection drops the same `@line:path:start-end` ref the
  // gutter drag produces — so the keyboard path mirrors dragging the lines into
  // the composer. Capture-phase + stopPropagation so it beats the terminal's
  // global ⌘L handler (which would otherwise grab the native text selection).
  useEffect(() => {
    if (!selection) {
      return
    }

    const onKeyDown = (event: KeyboardEvent) => {
      if (!isAddSelectionShortcut(event)) {
        return
      }

      const lineEnd = selection.end > selection.start ? selection.end : undefined
      const ref = droppedFileInlineRef({ line: selection.start, lineEnd, path: filePath }, $currentCwd.get())

      if (!ref) {
        return
      }

      event.preventDefault()
      event.stopPropagation()
      requestComposerInsertRefs([ref])
      requestComposerFocus('main')
    }

    window.addEventListener('keydown', onKeyDown, { capture: true })

    return () => window.removeEventListener('keydown', onKeyDown, { capture: true })
  }, [filePath, selection])

  return (
    <div className="grid w-full grid-cols-[auto_minmax(0,1fr)] font-mono text-xs leading-relaxed overflow-x-auto">
      <div className="select-none py-3 text-right text-muted-foreground/55">
        {Array.from({ length: lineCount }, (_, index) => {
          const line = index + 1
          const selected = inSelection(line)

          return (
            <div
              className={cn(
                'cursor-pointer px-3 tabular-nums transition-colors',
                selected
                  ? 'bg-amber-200/45 text-amber-900 dark:bg-amber-300/20 dark:text-amber-100'
                  : 'hover:text-foreground'
              )}
              draggable
              key={line}
              onClick={event => handleLineClick(event, line)}
              onDragStart={event => handleDragStart(event, line)}
              title={t.preview.sourceLineTitle}
            >
              {line}
            </div>
          )
        })}
      </div>
      <div
        className="relative [&_pre]:m-0 [&_pre]:px-3 [&_pre]:py-3 [&_pre]:bg-transparent! [&_pre]:whitespace-pre-wrap [&_pre]:break-words"
        data-selectable-text="true"
      >
        {selection && (
          <div
            aria-hidden
            className="pointer-events-none absolute inset-x-0 bg-amber-200/35 dark:bg-amber-300/10"
            style={{
              top: `calc(${SOURCE_PAD_Y_REM}rem + ${selection.start - 1} * ${SOURCE_LINE_HEIGHT_REM}rem)`,
              height: `calc(${selection.end - selection.start + 1} * ${SOURCE_LINE_HEIGHT_REM}rem)`
            }}
          />
        )}
        <div
          aria-hidden={editorVisible}
          className={cn(
            'transition-opacity duration-200 ease-out',
            editorVisible ? 'pointer-events-none opacity-0' : 'opacity-100'
          )}
        >
          <HighlightedSource key={text} language={language} onReady={handleHighlightedReady} text={text} />
        </div>
        <textarea
          aria-hidden={!editorVisible}
          className={cn(
            'absolute inset-0 resize-none border-0 bg-background/96 p-3 font-mono text-xs leading-relaxed text-foreground outline-none whitespace-pre-wrap transition-opacity duration-200 ease-out',
            editorVisible ? 'opacity-100' : 'pointer-events-none opacity-0'
          )}
          disabled={status === 'saving'}
          onChange={event => setEditContent(event.target.value)}
          ref={editorRef}
          spellCheck={false}
          value={editContent}
        />
        <div className="absolute top-1 right-2 z-10">
          <button
            className="min-w-11 rounded bg-background/80 px-2 py-0.5 text-[0.625rem] font-bold text-muted-foreground shadow-sm backdrop-blur transition-colors hover:text-foreground disabled:cursor-wait disabled:opacity-70"
            disabled={status === 'saving'}
            onClick={status === 'view' ? () => setStatus('editing') : () => void handleSave()}
            type="button"
          >
            {status === 'saving' ? 'Saving…' : status === 'editing' ? 'Save' : 'Edit'}
          </button>
        </div>
      </div>
    </div>
  )
}

export function LocalFilePreview({ reloadKey, target }: { reloadKey: number; target: PreviewTarget }) {
  const { t } = useI18n()
  const [state, setState] = useState<LocalPreviewState>({ loading: true })
  const [forcePreview, setForcePreview] = useState(false)
  const [renderMarkdownAsSource, setRenderMarkdownAsSource] = useState(false)
  const [renderMode, setRenderMode] = useState<RenderMode>(target.renderMode === 'diff' ? 'diff' : 'source')
  const filePath = filePathForTarget(target)
  const isImage = target.previewKind === 'image'

  const [gitDiff, setGitDiff] = useState<{ diff: string; status: string; fileContent: string; headContent: string }>({
    diff: '',
    status: '',
    fileContent: '',
    headContent: ''
  })

  // Fetch git diff when file loads or reloadKey changes
  useEffect(() => {
    if (!filePath || isImage) {
      return
    }

    let active = true

    const fetchDiff = async () => {
      try {
        const result = await window.hermesDesktop?.gitFileDiff?.(filePath, target.gitOriginalPath)

        if (active && result) {
          setGitDiff(result)
        }
      } catch {
        if (active) {
          setGitDiff({ diff: '', status: '', fileContent: '', headContent: '' })
        }
      }
    }

    void fetchDiff()

    return () => {
      active = false
    }
  }, [filePath, isImage, reloadKey, target.gitOriginalPath])

  useEffect(() => {
    if (target.renderMode === 'diff') {
      setRenderMode('diff')
    } else if (target.renderMode === 'source') {
      setRenderMode('source')
    } else if (target.renderMode === 'preview') {
      setRenderMode('preview')
    }
  }, [target.renderMode, target.url])

  const hasDiff = Boolean(gitDiff.status || gitDiff.headContent)

  // HTML files are rendered as source code, not in a webview - so they take
  // the same path as plain text files. `previewKind === 'binary'` arrives
  // when the file is forcibly previewed past the binary refusal screen.
  const isText = target.previewKind === 'text' || target.previewKind === 'binary' || target.previewKind === 'html'

  const blockedByTarget = !isImage && !forcePreview && (target.binary || target.large)

  useEffect(() => {
    let active = true

    async function load() {
      if (blockedByTarget) {
        setState({ loading: false })

        return
      }

      if (!isImage && !isText) {
        setState({ loading: false })

        return
      }

      setState({ loading: true })

      try {
        if (isImage) {
          // Prefer bytes the caller already handed us (a pasted/dropped
          // screenshot) over re-reading a path that may be transient/unreadable.
          const dataUrl = target.dataUrl || (await readDesktopFileDataUrl(filePath))

          if (active) {
            setState({ dataUrl, loading: false })
          }

          return
        }

        const result = await readTextPreview(filePath)

        if (active) {
          const shouldBlock = !forcePreview && (result.binary || (result.byteSize ?? 0) > TEXT_PREVIEW_MAX_BYTES)

          setState({
            binary: result.binary,
            byteSize: result.byteSize,
            language: result.language || target.language || 'text',
            loading: false,
            text: shouldBlock ? undefined : result.text,
            truncated: result.truncated
          })
        }
      } catch (error) {
        if (active) {
          setState({
            error: error instanceof Error ? error.message : String(error),
            loading: false
          })
        }
      }
    }

    void load()

    return () => {
      active = false
    }
  }, [blockedByTarget, filePath, forcePreview, isImage, isText, reloadKey, target.dataUrl, target.language])

  if (state.loading) {
    return <PageLoader label={t.preview.loading} />
  }

  if (state.error && renderMode !== 'diff') {
    return <PreviewEmptyState body={state.error} title={t.preview.unavailable} />
  }

  const isMarkdown = (state.language || target.language) === 'markdown'
  const showRendered = isMarkdown && !renderMarkdownAsSource && renderMode !== 'source' && renderMode !== 'diff'
  const hasTextContent = isText && state.text !== undefined

  // Binary/large block — only when not force-previewed and not viewing diff
  if (
    !isImage &&
    !forcePreview &&
    renderMode !== 'diff' &&
    (target.binary || target.large || state.binary || (state.byteSize ?? 0) > TEXT_PREVIEW_MAX_BYTES)
  ) {
    const binary = target.binary || state.binary
    const size = target.byteSize || state.byteSize

    return (
      <div className="h-full overflow-auto bg-transparent">
        <PreviewToggle
          hasDiff={hasDiff}
          hasPreview={isMarkdown}
          mode={renderMode}
          onModeChange={newMode => {
            setRenderMode(newMode)

            if (newMode === 'source') {
              setRenderMarkdownAsSource(true)
            } else if (newMode === 'preview') {
              setRenderMarkdownAsSource(false)
            }
          }}
        />
        <PreviewEmptyState
          body={binary ? t.preview.binaryBody(target.label) : t.preview.largeBody(target.label, formatBytes(size))}
          primaryAction={{ label: t.preview.previewAnyway, onClick: () => setForcePreview(true) }}
          title={binary ? t.preview.binaryTitle : t.preview.largeTitle}
          tone="warning"
        />
      </div>
    )
  }

  if (isImage && state.dataUrl) {
    return (
      <div className="h-full overflow-auto bg-transparent">
        <PreviewToggle
          hasDiff={hasDiff}
          hasPreview={isMarkdown}
          mode={renderMode}
          onModeChange={newMode => {
            setRenderMode(newMode)

            if (newMode === 'source') {
              setRenderMarkdownAsSource(true)
            } else if (newMode === 'preview') {
              setRenderMarkdownAsSource(false)
            }
          }}
        />
        <div className="flex h-[calc(100%-2rem)] w-full items-center justify-center overflow-auto p-4">
          <img
            alt={target.label}
            className="max-h-full max-w-full rounded-lg object-contain shadow-sm"
            draggable={false}
            src={state.dataUrl}
          />
        </div>
      </div>
    )
  }

  if (renderMode === 'diff') {
    return (
      <div className="h-full overflow-auto bg-transparent select-text" data-selectable-text="true">
        <PreviewToggle
          hasDiff={hasDiff}
          hasPreview={isMarkdown}
          mode={renderMode}
          onModeChange={newMode => {
            setRenderMode(newMode)

            if (newMode === 'source') {
              setRenderMarkdownAsSource(true)
            } else if (newMode === 'preview') {
              setRenderMarkdownAsSource(false)
            }
          }}
        />
        {hasDiff ? (
          <DiffView fileContent={gitDiff.fileContent || state.text || ''} headContent={gitDiff.headContent || ''} />
        ) : (
          <div className="grid h-32 place-items-center text-xs text-muted-foreground/60">No uncommitted changes</div>
        )}
      </div>
    )
  }

  if (hasTextContent) {
    return (
      <div className="h-full overflow-auto bg-transparent select-text" data-selectable-text="true">
        <PreviewToggle
          hasDiff={hasDiff}
          hasPreview={isMarkdown}
          mode={renderMode}
          onModeChange={newMode => {
            setRenderMode(newMode)

            if (newMode === 'source') {
              setRenderMarkdownAsSource(true)
            } else if (newMode === 'preview') {
              setRenderMarkdownAsSource(false)
            }
          }}
        />
        {state.truncated && (
          <div className="border-b border-border/60 bg-muted/35 px-3 py-1.5 text-[0.68rem] text-muted-foreground">
            {t.preview.truncated}
          </div>
        )}
        {showRendered ? (
          <MarkdownPreview text={state.text!} />
        ) : (
          <SourceView
            filePath={filePath}
            language={state.language || 'text'}
            onContentSaved={newContent => {
              setState(prev => ({ ...prev, text: newContent }))
              setGitDiff(prev => ({ ...prev, fileContent: newContent }))
            }}
            text={state.text!}
          />
        )}
      </div>
    )
  }

  return (
    <div className="h-full overflow-auto bg-transparent select-text" data-selectable-text="true">
      <PreviewToggle
        hasDiff={hasDiff}
        hasPreview={isMarkdown}
        mode={renderMode}
        onModeChange={newMode => {
          setRenderMode(newMode)

          if (newMode === 'source') {
            setRenderMarkdownAsSource(true)
          } else if (newMode === 'preview') {
            setRenderMarkdownAsSource(false)
          }
        }}
      />
      <PreviewEmptyState body={t.preview.noInlineBody(target.mimeType || '')} title={t.preview.noInlineTitle} />
    </div>
  )
}

import { useStore } from '@nanostores/react'
import type * as React from 'react'
import type {
  ComponentProps,
  CSSProperties,
  DragEvent as ReactDragEvent,
  MouseEvent as ReactMouseEvent,
  ReactNode
} from 'react'
import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Streamdown } from 'streamdown'

import { requestComposerFocus, requestComposerInsertRefs } from '@/app/chat/composer/focus'
import { droppedFileInlineRef } from '@/app/chat/composer/inline-refs'
import { HERMES_PATHS_MIME } from '@/app/chat/hooks/use-composer-actions'
import { isAddSelectionShortcut } from '@/app/right-sidebar/terminal/selection'
import { RichCodeBlock } from '@/components/assistant-ui/embeds'
import { CodeEditor } from '@/components/chat/code-editor'
import { FileDiffPanel } from '@/components/chat/diff-lines'
import { chunkTextLines, useFixedRowWindow } from '@/components/chat/fixed-row-window'
import { LazyShiki as ShikiHighlighter } from '@/components/chat/shiki-highlighter'
import { PageLoader } from '@/components/page-loader'
import { Tip } from '@/components/ui/tooltip'
import { translateNow, useI18n } from '@/i18n'
import {
  desktopFileDiff,
  desktopFsCacheKey,
  desktopGitRoot,
  readDesktopFileDataUrl,
  readDesktopFileText,
  writeDesktopFileText
} from '@/lib/desktop-fs'
import { Check, Pencil, X } from '@/lib/icons'
import { createMemoizedMathPlugin } from '@/lib/katex-memo'
import { shikiLanguageForFilename } from '@/lib/markdown-code'
import { normalizeFilePreviewMath } from '@/lib/markdown-preprocess'
import { cn } from '@/lib/utils'
import type { PreviewTarget } from '@/store/preview'
import { setPreviewDirty } from '@/store/preview-edit'
import { $connection, $currentCwd } from '@/store/session'
import { notifyWorkspaceChanged } from '@/store/workspace-events'

const SHIKI_THEME = { dark: 'github-dark-default', light: 'github-light-default' } as const
const TEXT_PREVIEW_MAX_BYTES = 512 * 1024
const SOURCE_CHUNK_LINES = 200
const SOURCE_LINE_PX = 20
const SOURCE_OVERSCAN_LINES = 400

// Math plugin for the static file preview, configured once at module scope.
// Mirrors the chat transcript's plugin (`markdown-text.tsx`) — same memoized
// KaTeX wrapper, with `singleDollarTextMath: true` so `$x$` renders inline.
const previewMathPlugin = createMemoizedMathPlugin({ singleDollarTextMath: true })

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
  /** Working-tree-vs-HEAD unified diff, when the file has uncommitted changes. */
  diff?: string
  error?: string
  language?: string
  loading: boolean
  text?: string
  truncated?: boolean
}

// True when focus is in a field that should swallow plain keystrokes (so the
// bare-`e` edit shortcut never fires while the user is typing in the composer,
// a search box, or the editor itself).
function isTypableElement(el: Element | null): boolean {
  if (!el) {
    return false
  }

  const tag = el.tagName

  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || (el as HTMLElement).isContentEditable
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

function dataUrlToBlob(dataUrl: string) {
  const comma = dataUrl.indexOf(',')

  if (comma < 0 || !dataUrl.startsWith('data:')) {
    throw new Error('Invalid PDF data URL')
  }

  const metadata = dataUrl
    .slice(5, comma)
    .split(';')
    .map(part => part.trim().toLowerCase())

  const payload = dataUrl.slice(comma + 1)

  if (metadata[0] !== 'application/pdf' || !metadata.slice(1).includes('base64')) {
    throw new Error('Invalid PDF data URL type')
  }

  let binary: string

  try {
    binary = atob(decodeURIComponent(payload))
  } catch {
    throw new Error('Invalid PDF data URL payload')
  }

  if (!binary.startsWith('%PDF-')) {
    throw new Error('Invalid PDF file header')
  }

  const bytes = new Uint8Array(binary.length)

  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index)
  }

  return new Blob([bytes], { type: 'application/pdf' })
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
  pre: 'mb-4 overflow-hidden rounded-lg border border-border bg-card font-mono text-xs leading-relaxed last:mb-0 [&_pre]:m-0 [&_pre]:overflow-x-auto [&_pre]:bg-transparent! [&_pre]:p-3 [&_pre]:font-mono',
  hr: 'my-6 border-border',
  th: 'px-3 py-2 text-left text-sm font-semibold text-foreground',
  td: 'px-3 py-2 align-top text-sm leading-relaxed',
  thead: 'bg-muted/35 text-muted-foreground'
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

  const code = String(children).replace(/\n$/, '')

  const highlighted = (
    <ShikiHighlighter
      addDefaultStyles={false}
      as="div"
      defaultColor="light-dark()"
      delay={80}
      language={language}
      showLanguage={false}
      theme={SHIKI_THEME}
    >
      {code}
    </ShikiHighlighter>
  )

  // ```mermaid / ```svg fences route to the shared lazy renderers (same
  // registry the chat transcript uses); everything else stays on Shiki.
  return <RichCodeBlock code={code} fallback={highlighted} language={language} />
}

function MarkdownTable({ className, ...rest }: ComponentProps<'table'>) {
  return (
    <div className="mb-4 w-full overflow-x-auto rounded-lg border border-border last:mb-0">
      <table
        className={cn(
          'm-0 w-full min-w-[18rem] border-collapse [&_tr]:border-b [&_tr]:border-border last:[&_tr]:border-0',
          className
        )}
        {...rest}
      />
    </div>
  )
}

function MarkdownImage({ alt, src, ...rest }: ComponentProps<'img'>) {
  return (
    <img
      alt={alt ?? ''}
      className="my-3 max-h-96 w-auto max-w-full rounded-lg border border-border object-contain shadow-sm"
      src={src}
      {...rest}
    />
  )
}

function MarkdownLink({ children, className, href, ...rest }: ComponentProps<'a'>) {
  const isExternal = /^https?:\/\//i.test(href || '')

  return (
    <a
      className={cn('text-foreground underline underline-offset-2 hover:text-primary', className)}
      href={href}
      rel={isExternal ? 'noopener noreferrer' : undefined}
      target={isExternal ? '_blank' : undefined}
      {...rest}
    >
      {children}
    </a>
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
  code: MarkdownCode,
  hr: tagged('hr'),
  table: MarkdownTable,
  th: tagged('th'),
  td: tagged('td'),
  thead: tagged('thead'),
  img: MarkdownImage,
  a: MarkdownLink
}
// --- Reading-position preservation (rendered markdown view) ---------------
// A raw scrollTop is the wrong anchor when an edit inserts or removes lines
// above the viewport: the same pixel offset then points at different text.
// Instead we capture WHAT the user was reading (text of the first visible
// block + how far into it the viewport top sits) and after the reload search
// the new DOM for that text. Shorter prefix fallbacks tolerate small edits
// inside the anchored block itself; taking the topmost match biases to the
// nearest unchanged block above, which lands the viewport on the change.

interface ScrollAnchor {
  fraction: number
  prefix: string
  scrollTop: number
}

const ANCHOR_SELECTOR = 'p, li, h1, h2, h3, h4, pre, blockquote, tr'

function normalizeAnchorText(value: string | null) {
  return (value || '').replace(/\s+/g, ' ').trim()
}

function captureScrollAnchor(scroller: HTMLElement): ScrollAnchor | null {
  const viewportTop = scroller.getBoundingClientRect().top
  const blocks = Array.from(scroller.querySelectorAll<HTMLElement>(ANCHOR_SELECTOR))

  for (const block of blocks) {
    const rect = block.getBoundingClientRect()

    if (rect.bottom <= viewportTop + 1) {
      continue
    }

    const text = normalizeAnchorText(block.textContent)

    if (!text) {
      continue
    }

    const fraction =
      rect.height > 0 ? Math.min(1, Math.max(0, (viewportTop - rect.top) / rect.height)) : 0

    return { fraction, prefix: text.slice(0, 80), scrollTop: scroller.scrollTop }
  }

  return null
}

function restoreScrollAnchor(scroller: HTMLElement, anchor: ScrollAnchor): boolean {
  const blocks = Array.from(scroller.querySelectorAll<HTMLElement>(ANCHOR_SELECTOR))
  const prefixes = [anchor.prefix, anchor.prefix.slice(0, 40), anchor.prefix.slice(0, 20)]
  const scrollerTop = scroller.getBoundingClientRect().top

  for (const prefix of prefixes) {
    if (!prefix) {
      continue
    }

    for (const block of blocks) {
      if (!normalizeAnchorText(block.textContent).startsWith(prefix)) {
        continue
      }

      const rect = block.getBoundingClientRect()
      const blockTop = rect.top - scrollerTop + scroller.scrollTop
      const target = blockTop + anchor.fraction * rect.height
      const max = Math.max(0, scroller.scrollHeight - scroller.clientHeight)

      scroller.scrollTop = Math.min(target, max)

      return true
    }
  }

  return false
}

// First line (1-based, in the new text) where old and new differ - a plain
// common-prefix walk, no git needed.
function firstChangedLine(oldText: string, newText: string): number {
  const oldLines = oldText.split('\n')
  const newLines = newText.split('\n')
  const maxPrefix = Math.min(oldLines.length, newLines.length)
  let prefix = 0

  while (prefix < maxPrefix && oldLines[prefix] === newLines[prefix]) {
    prefix += 1
  }

  return Math.min(prefix + 1, Math.max(1, newLines.length))
}

export interface ChangedLineSpan {
  /** First changed line (1-based, in the NEW text); null when only lines
   * were deleted (the change has no new-text position). */
  line: number | null
  /** Last changed/new line of the span, inclusive (1-based, new text). */
  endLine: number
  /** True when the span contains at least one inserted/modified line (a
   * place the preview can scroll TO); pure deletions only mark context. */
  hasNew: boolean
}

/** Changed regions between two texts, as consecutive 1-based line spans in
 *  the NEW text. A plain common-prefix + common-suffix walk sandwiches the
 *  differing middle; runs of blank separator lines inside it split the
 *  middle into separate spans so the jump button can cycle change 1, 2, 3…
 *  Pure-deletion gaps between insertions are merged into the nearest span
 *  rather than emitted as unjumpable noise. */
export function changedLineSpans(oldText: string, newText: string): ChangedLineSpan[] {
  const oldLines = oldText.split('\n')
  const newLines = newText.split('\n')
  const maxPrefix = Math.min(oldLines.length, newLines.length)
  let prefix = 0

  while (prefix < maxPrefix && oldLines[prefix] === newLines[prefix]) {
    prefix += 1
  }

  const maxSuffix = Math.min(oldLines.length - prefix, newLines.length - prefix)
  let suffix = 0

  while (suffix < maxSuffix && oldLines[oldLines.length - 1 - suffix] === newLines[newLines.length - 1 - suffix]) {
    suffix += 1
  }

  // 1-based, inclusive [start, end] of the differing middle in the NEW text.
  const start = prefix + 1
  const end = newLines.length - suffix

  if (start > end) {
    return []
  }

  // Split the middle on runs of >=2 blank lines (paragraph boundaries).
  const spans: { end: number; start: number }[] = []
  let spanStart = 0
  let blanks = 0

  for (let i = start; i <= end; i += 1) {
    if (!newLines[i - 1].trim()) {
      blanks += 1

      continue
    }

    // Non-blank line after a blank separator: close the pending span. A
    // markdown paragraph break is exactly ONE empty line in the array.
    if (blanks >= 1 && spanStart > 0) {
      spans.push({ end: i - 1 - blanks, start: spanStart })
      spanStart = 0
    }

    blanks = 0

    if (spanStart === 0) {
      spanStart = i
    }
  }

  if (spanStart > 0 && spanStart <= end) {
    spans.push({ end, start: spanStart })
  }

  // Mark each span by whether it contains at least one new-text line that
  // differs from the old text at the aligned position. Cheap alignment: a
  // line is "new" when it does not appear anywhere in the old middle.
  const oldMiddle = new Set(oldLines.slice(prefix, oldLines.length - suffix))

  return spans.map(span => {
    const hasNew = newLines
      .slice(span.start - 1, span.end)
      .some(line => !oldMiddle.has(line))

    return { endLine: span.end, hasNew, line: hasNew ? span.start : null }
  })
}

// ---- Cross-view (rendered preview <-> source) position mapping ----
// Rendered blocks and source lines share no coordinates, so mapping goes
// through normalized TEXT: a paragraph/heading/list line in the source is,
// after stripping markdown markers, the same words as the rendered block.
// Markdown whose rendered form has no shared text (tables, KaTeX math,
// frontmatter) degrades to a fractional scroll position instead of
// pretending to be exact - same philosophy as the reload scroll anchor.

/** Strip markdown syntax from a source line, keeping its visible words. */
export function normalizeForCrossView(raw: string): string {
  return raw
    .replace(/^#{1,6}\s+/, '')
    .replace(/^>\s?>?\s?/, '')
    .replace(/^[-*+]\s+/, '')
    .replace(/^\d+[.)]\s+/, '')
    .replace(/^\|\s*/, '')
    .replace(/\s*\|\s*/g, ' ')
    .replace(/[*_`~]/g, '')
    .replace(/\$/g, '')
    .replace(/\s+/g, ' ')
    .trim()
}

function usableCrossViewText(normalized: string): boolean {
  return normalized.length >= 8 && !normalized.startsWith('---')
}

export interface CrossViewAnchor {
  /** Overall document fraction (0..1) - the fallback when text matches. */
  fraction: number
  /** Normalized text prefix of the anchor block/line, or null when the
   * surrounding markdown has no text that renders identically. */
  prefix: string | null
  /** Source line captured with the anchor (source-side captures only) -
   * lets a later source-side restore skip text matching entirely. */
  line?: number | null
}

/** Find a source line near `line` (walking outward) whose visible text can
 * anchor a cross-view match. Returns a fraction fallback either way. */
export function anchorNearLine(lines: string[], line: number): CrossViewAnchor {
  const total = lines.length
  const fallbackFraction = Math.min(1, Math.max(0, (line - 1) / Math.max(1, total - 1)))

  for (let radius = 0; radius <= 12; radius += 1) {
    for (const candidate of radius === 0 ? [line] : [line + radius, line - radius]) {
      if (candidate < 1 || candidate > total) {
        continue
      }

      const normalized = normalizeForCrossView(lines[candidate - 1])

      if (usableCrossViewText(normalized)) {
        return {
          fraction: Math.min(1, Math.max(0, (candidate - 1) / Math.max(1, total - 1))),
          prefix: normalized.slice(0, 80)
        }
      }
    }
  }

  return { fraction: fallbackFraction, prefix: null }
}

/** Find the first source line whose visible text matches `prefix` (the
 * rendered-block text walks backwards through the same normalization). */
export function lineForCrossViewPrefix(lines: string[], prefix: string): number | null {
  const target = normalizeForCrossView(prefix).slice(0, 40)

  if (target.length < 8) {return null}

  for (let index = 0; index < lines.length; index += 1) {
    const normalized = normalizeForCrossView(lines[index])

    if (usableCrossViewText(normalized) && (normalized.startsWith(target) || normalized.includes(target.slice(0, 20)))) {
      return index + 1
    }
  }

  return null
}

/** Find the rendered preview block (inside the scroller) whose text matches
 *  `prefix` (a normalized source-line prefix). */
export function previewBlockForPrefix(scroller: HTMLElement, prefix: string): HTMLElement | null {
  const target = normalizeAnchorText(prefix).slice(0, 40)

  if (target.length < 8) {return null}

  const blocks = Array.from(scroller.querySelectorAll<HTMLElement>(ANCHOR_SELECTOR))

  for (const block of blocks) {
    const text = normalizeAnchorText(block.textContent)

    if (text.length >= 8 && (text.startsWith(target) || text.includes(target.slice(0, 20)))) {
      return block
    }
  }

  return null
}

/** Plain-text version of a copied selection from the rendered preview.
 *  KaTeX renders each equation twice: a visually-correct HTML tree and a
 *  hidden MathML tree that carries the original LaTeX in <annotation>.
 *  Chromium serializes that MathML into the plain-text clipboard with every
 *  symbol on its own line - pasting that into the composer gives the agent
 *  one-glyph-per-line garbage.
 *
 *  Walk the selection Range over the LIVE dom (not the cloned fragment - a
 *  partially-selected equation clones only half the tree, which no longer
 *  contains the annotation): plain text nodes are sliced precisely to the
 *  character offsets of the selection, while any .katex the selection
 *  touches is emitted whole as its LaTeX source (an equation has no
 *  character-level mapping back to its source, so "touched" is the finest
 *  possible granularity). Output is squeezed onto one line: every run of
 *  whitespace (newlines, tabs, ...) becomes a single space, LaTeX is
 *  emitted bare with no $ / $$ delimiters. */
export function selectionTextWithMath(root: HTMLElement): string {
  const doc = root.ownerDocument
  const selection = doc.getSelection()

  if (!selection || selection.rangeCount === 0 || selection.isCollapsed) {
    return ''
  }

  // Always walk the ranges (never selection.toString()): the walker gives
  // exact character slices, consistent block-boundary spaces, and works
  // identically whether or not the selection touches math.
  let out = ''

  for (let i = 0; i < selection.rangeCount; i++) {
    out += rangeTextWithMath(selection.getRangeAt(i))
  }

  return squeezeWhitespace(out)
}

function squeezeWhitespace(text: string): string {
  return text.replace(/\s+/g, ' ').trim()
}

function katexAncestorOf(node: Node): Element | null {
  let cur: Node | null = node

  while (cur) {
    if (cur instanceof Element && cur.classList.contains('katex')) {return cur}
    cur = cur.parentNode
  }

  return null
}

const BLOCK_TAGS = new Set(['P', 'LI', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'PRE', 'BLOCKQUOTE', 'TR', 'TABLE', 'UL', 'OL'])

function blockAncestorOf(node: Node): Element | null {
  let cur: Node | null = node.parentNode

  while (cur) {
    if (cur instanceof Element && BLOCK_TAGS.has(cur.tagName)) {return cur}
    cur = cur.parentNode
  }

  return null
}

/** One range -> text: precise slices for plain text nodes, whole-LaTeX for
 *  any touched .katex, a separator space whenever the walk crosses a block
 *  boundary (so two adjacent paragraphs don't fuse into one word). */
function rangeTextWithMath(range: Range): string {
  const doc = range.commonAncestorContainer.ownerDocument!
  const emitted = new Set<Element>()
  let out = ''
  let lastBlock: Element | null = null
  let lastEndedWithSpace = true

  const append = (chunk: string) => {
    if (!chunk) {return}

    if (out && !lastEndedWithSpace && !/^\s/.test(chunk)) {out += ' '}
    out += chunk
    lastEndedWithSpace = /\s$/.test(chunk)
  }

  const emitTextNode = (node: Text) => {
    const data = node.data
    let from = 0
    let to = data.length

    if (node === range.startContainer) {from = range.startOffset}

    if (node === range.endContainer) {to = range.endOffset}

    if (to <= from) {return}

    const block = blockAncestorOf(node)

    if (block !== lastBlock) {
      append(' ')
      lastBlock = block
    }

    append(data.slice(from, to))
  }

  const emitKatex = (katex: Element) => {
    if (emitted.has(katex)) {return}
    emitted.add(katex)

    const latex = katex.querySelector('annotation')?.textContent?.trim()

    append(' ')
    append(latex || (katex.textContent ?? '').trim())
    append(' ')
  }

  // Selection inside a single text node (commonAncestorContainer is a Text):
  // a TreeWalker never visits its own root, so handle it directly.
  const anc = range.commonAncestorContainer

  if (anc.nodeType === Node.TEXT_NODE) {
    const katex = katexAncestorOf(anc)

    if (katex) {
      emitKatex(katex)
    } else {
      emitTextNode(anc as Text)
    }

    return out
  }

  const walker = doc.createTreeWalker(anc, NodeFilter.SHOW_TEXT)
  let node: Node | null

  while ((node = walker.nextNode())) {
    if (!range.intersectsNode(node)) {continue}

    const textNode = node as Text
    const katex = katexAncestorOf(textNode)

    if (katex) {
      emitKatex(katex)

      continue
    }

    emitTextNode(textNode)
  }

  return out
}

function scrollPreviewBlockIntoView(scroller: HTMLElement, block: HTMLElement) {
  const scrollerTop = scroller.getBoundingClientRect().top
  const blockTop = block.getBoundingClientRect().top - scrollerTop + scroller.scrollTop
  const max = Math.max(0, scroller.scrollHeight - scroller.clientHeight)

  scroller.scrollTop = Math.min(Math.max(0, blockTop - 8), max)
}


export function MarkdownPreview({ text }: { text: string }) {
  const mathText = useMemo(() => normalizeFilePreviewMath(text), [text])

  return (
    <div className="preview-markdown mx-auto max-w-3xl px-4 py-3 text-sm text-foreground" data-selectable-text="true">
      <Streamdown
        components={MARKDOWN_COMPONENTS}
        controls={false}
        mode="static"
        parseIncompleteMarkdown={false}
        plugins={{ math: previewMathPlugin }}
      >
        {mathText}
      </Streamdown>
    </div>
  )
}

export function PreviewModeSwitcher({
  active,
  modes,
  onSelect,
  trailing
}: {
  active: PreviewViewMode
  modes: PreviewViewMode[]
  onSelect: (mode: PreviewViewMode) => void
  trailing?: ReactNode
}) {
  const { t } = useI18n()
  const showModes = modes.length > 1

  if (!showModes && !trailing) {
    return null
  }

  const label: Record<PreviewViewMode, string> = {
    diff: t.preview.diff,
    rendered: t.preview.renderedPreview,
    source: t.preview.source
  }

  return (
    // Fixed height so the header is byte-identical between read and edit modes —
    // swapping the trailing controls must never move the body below it.
    <div className="flex h-7 shrink-0 items-center justify-end gap-3 border-b border-border/40 px-3">
      {showModes &&
        modes.map(mode => (
          <button
            className={cn(
              'text-[0.625rem] font-bold underline-offset-4 transition-colors',
              mode === active
                ? 'text-foreground underline decoration-current/30'
                : 'text-muted-foreground hover:text-foreground'
            )}
            key={mode}
            onClick={() => onSelect(mode)}
            type="button"
          >
            {label[mode]}
          </button>
        ))}
      {trailing && <div className="flex items-center gap-1.5">{trailing}</div>}
    </div>
  )
}

// Cancel / Save controls rendered as the header's trailing slot (not a bar of
// their own) so edit mode reuses the read-mode header row verbatim.
function EditControls({
  dirty,
  onCancel,
  onSave,
  saving
}: {
  dirty: boolean
  onCancel: () => void
  onSave: () => void
  saving: boolean
}) {
  const { t } = useI18n()

  return (
    <>
      <button
        className="flex items-center gap-1 rounded-md px-1.5 text-[0.625rem] font-bold text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
        onClick={onCancel}
        type="button"
      >
        <X className="size-3" />
        {t.common.cancel}
      </button>
      <button
        className="flex items-center gap-1 rounded-md bg-primary px-2 py-0.5 text-[0.625rem] font-bold text-primary-foreground shadow-xs transition-opacity hover:opacity-90 disabled:opacity-50"
        disabled={!dirty || saving}
        onClick={onSave}
        type="button"
      >
        <Check className="size-3" />
        {saving ? t.common.saving : t.common.save}
      </button>
    </>
  )
}

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

/** Windowed, Shiki-highlighted source. The gutter's line selection produces a
 *  `path:line` composer ref, so it is inert without a `filePath` (artifact
 *  content has no path to reference lines against). */
export function SourceView({
  filePath,
  language,
  scrollNonce,
  scrollToLine,
  text
}: {
  filePath?: string
  language: string
  scrollNonce?: number
  scrollToLine?: number
  text: string
}) {
  const { t } = useI18n()
  const chunks = useMemo(() => chunkTextLines(text, SOURCE_CHUNK_LINES), [text])
  const lastChunk = chunks.at(-1)
  const totalLines = lastChunk ? lastChunk.start + lastChunk.lines.length : 0

  const { afterRows, beforeRows, endChunk, onScroll, scrollerRef, startChunk } = useFixedRowWindow({
    overscanRows: SOURCE_OVERSCAN_LINES,
    rowPx: SOURCE_LINE_PX,
    rowsPerChunk: SOURCE_CHUNK_LINES,
    totalRows: totalLines
  })

  const visibleChunks = chunks.slice(startChunk, endChunk + 1)
  const [selection, setSelection] = useState<LineSelection | null>(null)
  const inSelection = (line: number) => selection != null && line >= selection.start && line <= selection.end

  const handleLineClick = (event: ReactMouseEvent, line: number) => {
    if (!filePath) {
      return
    }

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
    if (!filePath) {
      return
    }

    startLineDrag(event, filePath, inSelection(line) && selection ? selection : { end: line, start: line })
  }

  // ⌘/Ctrl+L with a line selection drops the same `@line:path:start-end` ref the
  // gutter drag produces — so the keyboard path mirrors dragging the lines into
  // the composer. Capture-phase + stopPropagation so it beats the terminal's
  // global ⌘L handler (which would otherwise grab the native text selection).
  useEffect(() => {
    if (!selection || !filePath) {
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
      // Insert into and focus the SAME composer — 'active' — so a tile that owns
      // focus keeps it instead of the ref landing in a tile but main stealing focus.
      requestComposerInsertRefs([ref])
      requestComposerFocus('active')
    }

    window.addEventListener('keydown', onKeyDown, { capture: true })

    return () => window.removeEventListener('keydown', onKeyDown, { capture: true })
  }, [filePath, selection])

  // "Jump to change": scroll so `scrollToLine` sits a couple of context lines
  // below the viewport top. Rows are fixed-height (SOURCE_LINE_PX), so this is
  // exact. Fires on prop change; double rAF so layout has settled.
  useEffect(() => {
    if (!scrollToLine || scrollToLine < 1) {return}

    const target = Math.max(0, (scrollToLine - 3) * SOURCE_LINE_PX)
    let f1 = 0
    let f2 = 0
    f1 = requestAnimationFrame(() => {
      f2 = requestAnimationFrame(() => {
        if (scrollerRef.current) {
          scrollerRef.current.scrollTop = target
        }
      })
    })

    return () => {
      cancelAnimationFrame(f1)
      cancelAnimationFrame(f2)
    }
  }, [scrollToLine, scrollNonce, text])

  return (
    <div className="h-full overflow-auto" data-source-scroller onScroll={onScroll} ref={scrollerRef}>
      <div className="grid min-w-max grid-cols-[auto_minmax(0,1fr)] font-mono text-[0.7rem] leading-relaxed">
        {beforeRows > 0 && <div aria-hidden className="col-span-2" style={{ height: beforeRows * SOURCE_LINE_PX }} />}
        {visibleChunks.map(chunk => (
          <Fragment key={chunk.start}>
            <div className="select-none text-right text-muted-foreground/55">
              {chunk.lines.map((_lineText, offset) => {
                const line = chunk.start + offset + 1
                const selected = inSelection(line)

                return (
                  <div
                    className={cn(
                      'h-5 w-9 pr-2 leading-5 tabular-nums transition-colors',
                      filePath && 'cursor-pointer',
                      selected
                        ? 'bg-amber-200/45 text-amber-900 dark:bg-amber-300/20 dark:text-amber-100'
                        : filePath && 'hover:text-foreground'
                    )}
                    draggable={Boolean(filePath)}
                    key={line}
                    onClick={event => handleLineClick(event, line)}
                    onDragStart={event => handleDragStart(event, line)}
                    title={filePath ? t.preview.sourceLineTitle : undefined}
                  >
                    {line}
                  </div>
                )
              })}
            </div>
            <div className="preview-source-code min-w-0 [&_pre]:m-0" data-selectable-text="true">
              <ShikiHighlighter
                addDefaultStyles={false}
                as="div"
                defaultColor="light-dark()"
                delay={80}
                language={language || 'text'}
                showLanguage={false}
                theme={SHIKI_THEME}
              >
                {chunk.text}
              </ShikiHighlighter>
            </div>
          </Fragment>
        ))}
        {afterRows > 0 && <div aria-hidden className="col-span-2" style={{ height: afterRows * SOURCE_LINE_PX }} />}
      </div>
    </div>
  )
}

export type PreviewViewMode = 'diff' | 'rendered' | 'source'

export function LocalFilePreview({ reloadKey, target }: { reloadKey: number; target: PreviewTarget }) {
  const { t } = useI18n()
  const [state, setState] = useState<LocalPreviewState>({ loading: true })
  const [forcePreview, setForcePreview] = useState(false)
  const [pdfError, setPdfError] = useState<string>()
  const [pdfUrl, setPdfUrl] = useState<string>()
  // User-picked view; null = auto (diff when changed, else rendered markdown,
  // else source). Reset when the previewed file changes.
  const [userMode, setUserMode] = useState<null | PreviewViewMode>(null)
  // Spot-editor state. The editor owns its buffer (keyed by `editorKey`); the
  // live draft + the snapshot the user started from live in refs so typing
  // never re-renders this (large) component — `dirty` is the only render-worthy
  // signal and it flips just once when crossing the clean↔dirty boundary.
  // `selfReload` re-runs the load after a save without the parent.
  const [editing, setEditing] = useState(false)
  const draftRef = useRef('')
  const baselineRef = useRef('')
  const [dirty, setDirty] = useState(false)
  const [editorKey, setEditorKey] = useState(0)
  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState<null | string>(null)
  const [conflict, setConflict] = useState(false)
  const [selfReload, setSelfReload] = useState(0)

  // Where to jump when the user clicks "jump to change": ALL changed regions
  // (the jump button cycles through them, then back to the original reading
  // position) plus a click counter so each click re-fires the scroll, plus
  // per-change cross-view anchors (text prefix + fraction) so the rendered
  // preview can scroll to the same change without switching views. null =
  // nothing changed since open / last file switch.
  const [jumpTarget, setJumpTarget] = useState<{
    /** Changes in document order; jumps walk this list. */
    changes: { anchor: CrossViewAnchor; line: number }[]
    /** Next change index to jump to (0-based). When it reaches length, the
     * next click restores the original position and wraps to 0. */
    index: number
    nonce: number
    /** Where the user was when the change arrived - restored after the last
     * change in the cycle. In whichever scroller is live at restore time. */
    original: CrossViewAnchor | null
    originalScrollTop: number
  } | null>(null)

  // Line-exact scroll request consumed by SourceView (fires on nonce bump even
  // when the line is unchanged). Serves both "jump to change" (source view)
  // and rendered->source position sync.
  const [scrollRequest, setScrollRequest] = useState<{ line: number; nonce: number } | null>(null)
  // Position to restore in the view the user is switching TO (captured from
  // the view they are leaving). Cleared once applied.
  const pendingViewAnchorRef = useRef<CrossViewAnchor | null>(null)
  // For the bare-`e` shortcut: the read-view root (to detect focus-within) and a
  // hover flag (no state - only the keydown handler reads it).
  const readViewRef = useRef<HTMLDivElement>(null)
  const scrollerRef = useRef<HTMLDivElement>(null)
  const hoverRef = useRef(false)
  // Text-anchor scroll preservation: on reload we capture the text of the
  // first visible block + how far into it the viewport top sits, then after
  // the new content mounts we search for that text and restore the position.
  // A raw pixel scrollTop is the wrong anchor when the edit inserts or
  // removes lines above the viewport - the anchor tracks *what* the user was
  // reading, not *where*.
  const scrollAnchorRef = useRef<ScrollAnchor | null>(null)
  // Text shown before the latest reload - the "old" side of the line compare.
  const lastTextRef = useRef<string | undefined>(undefined)
  // Current view mode for async load()/restore paths (effect closures would
  // otherwise see a stale render's mode).
  const modeRef = useRef<PreviewViewMode | null>(null)
  const connection = useStore($connection)
  const fsCacheKey = desktopFsCacheKey(connection)
  const filePath = filePathForTarget(target)
  const isImage = target.previewKind === 'image'
  const isPdf = target.previewKind === 'pdf'

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    setUserMode(null)
    setEditing(false)
    setDirty(false)
    setSaving(false)
    setSaveError(null)
    setConflict(false)
    draftRef.current = ''
    baselineRef.current = ''
    // This effect now runs ONLY on a file switch (see deps), so wiping the
    // baseline here is correct again. An external reload (reloadKey bump from
    // the file watcher / manual refresh) must KEEP lastTextRef: the load
    // effect compares it against the new text to decide whether to show the
    // "jump to change" button. Wiping it on reloadKey raced ahead of that
    // compare and the button could never appear.
    lastTextRef.current = undefined
    scrollAnchorRef.current = null
    setJumpTarget(null)
    // A file switch must also drop the loaded state (text/dataUrl/diff):
    // the blob effect for PDFs keys on state.dataUrl, and a stale value that
    // happens to equal the new one (same bytes) would skip rebuilding the
    // object URL. Reload-triggered refreshes (reloadKey) deliberately do NOT
    // come through here - they keep showing the old content while reloading.
    setState({ loading: true })
  }, [filePath])

  // HTML files are rendered as source code, not in a webview - so they take
  // the same path as plain text files. `previewKind === 'binary'` arrives
  // when the file is forcibly previewed past the binary refusal screen.
  const isText = target.previewKind === 'text' || target.previewKind === 'binary' || target.previewKind === 'html'

  const blockedByTarget = !isImage && !isPdf && !forcePreview && (target.binary || target.large)

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    let active = true

    async function load() {
      if (blockedByTarget) {
        setState({ loading: false })

        return
      }

      if (!isImage && !isPdf && !isText) {
        setState({ loading: false })

        return
      }

      setState({ loading: true })

      try {
        if (isImage || isPdf) {
          // Prefer bytes the caller already handed us (a pasted/dropped
          // screenshot) over re-reading a path that may be transient/unreadable.
          const dataUrl = target.dataUrl || (await readDesktopFileDataUrl(filePath))

          if (active) {
            setState({ dataUrl, loading: false })
          }

          return
        }

        // Capture what the user is reading (text anchor, not pixels) before the
        // new content replaces the old, and remember the old text so the
        // "jump to change" button can compute the first differing line.
        if (scrollerRef.current) {
          scrollAnchorRef.current = captureScrollAnchor(scrollerRef.current)
        }

        const result = await readTextPreview(filePath)

        if (active) {
          const shouldBlock = !forcePreview && (result.binary || (result.byteSize ?? 0) > TEXT_PREVIEW_MAX_BYTES)

          // Old text for the change-detect (jump button). Skip the very first
          // load (nothing was shown yet) and file switches (reset effect
          // clears the ref). Read from a ref, not closure state - this effect
          // closure can be older than the latest render.
          const previous = lastTextRef.current

          if (previous !== undefined && previous !== result.text) {
            const newLines = result.text.split('\n')

            const spans = changedLineSpans(previous, result.text)
              .filter(span => span.line !== null)
              .map(span => ({
                anchor: anchorNearLine(newLines, span.line as number),
                line: span.line as number
              }))

            // Fall back to the first differing line when the span walk
            // found nothing jumpable (pure deletions at the tail, etc.).
            if (spans.length === 0) {
              const line = firstChangedLine(previous, result.text)

              spans.push({ anchor: anchorNearLine(newLines, line), line })
            }

            // Where the user is right now becomes the "restore" target after
            // the last change in the cycle.
            const currentScroller = scrollerRef.current

            const original =
              modeRef.current === 'rendered'
                ? (currentScroller ? captureScrollAnchor(currentScroller) : null)
                : null

            setJumpTarget({
              changes: spans,
              index: 0,
              nonce: 0,
              original: original
                ? {
                    fraction:
                      original.scrollTop /
                      Math.max(1, currentScroller!.scrollHeight - currentScroller!.clientHeight),
                    prefix: original.prefix
                  }
                : null,
              originalScrollTop: currentScroller ? currentScroller.scrollTop : 0
            })
          }

          setState({
            binary: result.binary,
            byteSize: result.byteSize,
            language: result.language || target.language || 'text',
            loading: false,
            text: shouldBlock ? undefined : result.text,
            truncated: result.truncated
          })

          if (!shouldBlock) {
            lastTextRef.current = result.text
          }

          // Best-effort: fetch the file's working-tree-vs-HEAD diff so the
          // preview can offer a DIFF view when there are uncommitted changes.
          // Empty (clean file / not a repo / remote) just hides the option.
          if (!shouldBlock) {
            try {
              const root = await desktopGitRoot(filePath)
              const diff = root ? await desktopFileDiff(root, filePath) : ''

              if (active && diff.trim()) {
                setState(prev => (prev.text === result.text ? { ...prev, diff } : prev))
              }
            } catch {
              // No diff available; the preview just shows source.
            }
          }
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
  }, [
    blockedByTarget,
    filePath,
    forcePreview,
    fsCacheKey,
    isImage,
    isPdf,
    isText,
    reloadKey,
    selfReload,
    target.dataUrl,
    target.language
  ])

  useEffect(() => {
    setPdfUrl(undefined)
    setPdfError(undefined)

    if (!isPdf || !state.dataUrl) {
      return
    }

    // Chromium's PDF viewer is blank for large data: URLs in an iframe. Use a
    // blob URL instead, and revoke it when the target or loaded bytes change.
    if (typeof URL.createObjectURL !== 'function') {
      setPdfError('PDF preview requires object URL support')

      return
    }

    let objectUrl: string

    try {
      objectUrl = URL.createObjectURL(dataUrlToBlob(state.dataUrl))
      setPdfUrl(objectUrl)
    } catch (error) {
      setPdfError(error instanceof Error ? error.message : String(error))

      return
    }

    return () => URL.revokeObjectURL(objectUrl)
  }, [isPdf, state.dataUrl])

  // Restore the reading position after a content reload. The anchor holds the
  // text of the first visible block (not pixels), so inserted/deleted lines
  // above the viewport no longer shift what the user was reading. Runs after
  // the new content paints (two rAFs); falls back to the old pixel scrollTop
  // when the anchored text is gone entirely (heavy rewrite around the user).
  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    if (state.loading) {return}
    const anchor = scrollAnchorRef.current

    if (!anchor) {return}
    scrollAnchorRef.current = null

    let f1 = 0
    let f2 = 0
    f1 = requestAnimationFrame(() => {
      f2 = requestAnimationFrame(() => {
        const scroller = scrollerRef.current

        if (!scroller) {return}

        const restored = restoreScrollAnchor(scroller, anchor)

        if (!restored) {
          const max = Math.max(0, scroller.scrollHeight - scroller.clientHeight)
          scroller.scrollTop = Math.min(anchor.scrollTop, max)
        }
      })
    })

    return () => {
      cancelAnimationFrame(f1)
      cancelAnimationFrame(f2)
    }
  }, [state.loading, state.text])

  // ---- View-switch position sync (rendered <-> source) ----
  // When the user switches views, capture where they were (as a CrossViewAnchor
  // - normalized text prefix + fraction) from the view they leave, and restore
  // it in the view they enter. Same text-matching philosophy as the reload
  // anchor: exact for text-bearing blocks, fractional fallback otherwise.

  /** The preview block currently at the top of the viewport, as an anchor. */
  const capturePreviewAnchor = useCallback((): CrossViewAnchor | null => {
    const scroller = scrollerRef.current

    if (!scroller || !scroller.scrollHeight) {return null}

    const captured = captureScrollAnchor(scroller)

    if (!captured) {return null}

    return { fraction: captured.scrollTop / Math.max(1, scroller.scrollHeight - scroller.clientHeight), prefix: captured.prefix }
  }, [])

  /** The source line currently at the top of the viewport, as an anchor. */
  const captureSourceAnchor = useCallback((): CrossViewAnchor | null => {
    // The source view scrolls inside its OWN container (nested one level
    // below the outer preview scroller, which stays at scrollTop 0 there).
    // Reading the outer one made every source->rendered switch land at the
    // document top. Prefer the tagged inner container; fall back for safety.
    const outer = scrollerRef.current
    const inner = outer?.querySelector<HTMLElement>('[data-source-scroller]') ?? outer
    const text = state.text

    if (!inner || !text) {return null}

    const lines = text.split('\n')
    const topLine = Math.floor(inner.scrollTop / SOURCE_LINE_PX) + 1
    const safeTop = Math.min(topLine, lines.length)

    return { ...anchorNearLine(lines, safeTop), line: safeTop }
  }, [state.text])

  /** Apply a pending anchor to whichever view just became visible. */
  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    const anchor = pendingViewAnchorRef.current

    if (!anchor || state.loading || !state.text) {return}
    pendingViewAnchorRef.current = null

    let f1 = 0
    let f2 = 0
    f1 = requestAnimationFrame(() => {
      f2 = requestAnimationFrame(() => {
        const scroller = scrollerRef.current

        if (!scroller) {return}

        if (modeRef.current === 'rendered') {
          const block = anchor.prefix ? previewBlockForPrefix(scroller, anchor.prefix) : null

          if (block) {
            scrollPreviewBlockIntoView(scroller, block)
          } else {
            const max = Math.max(0, scroller.scrollHeight - scroller.clientHeight)
            scroller.scrollTop = anchor.fraction * max
          }
        } else if (modeRef.current === 'source') {
          const line = anchor.prefix ? lineForCrossViewPrefix((state.text ?? '').split('\n'), anchor.prefix) : null

          // Text match failed (math/table-heavy region) or no prefix: fall
          // back to the fractional position mapped onto line count. Without
          // this the freshly remounted source view just sat at the top and
          // the second switch looked like "positioning failed".
          const fallback =
            anchor.fraction > 0 || anchor.line != null
              ? Math.max(1, Math.round(anchor.fraction * ((state.text ?? '').split('\n').length - 2)) + 1)
              : null

          const target = line ?? anchor.line ?? fallback

          if (target) {
            setScrollRequest({ line: target, nonce: Date.now() })
          }
        }
      })
    })

    return () => {
      cancelAnimationFrame(f1)
      cancelAnimationFrame(f2)
    }
    // Fires when the user picks a view (userMode) or content settles; reads
    // the live mode from modeRef. modeRef is a ref - always current.
     
  }, [userMode, state.loading, state.text])

  // Editing is only offered for whole, readable text — never images, binaries,
  // or files we only loaded the first 512 KB of (saving would drop the tail).
  const canEdit =
    isText && !isImage && !blockedByTarget && state.text !== undefined && !state.truncated && !state.binary

  // Per-keystroke: update the draft ref (no render) and only set `dirty` when it
  // actually changes — React bails on an identical value, so a long typing run
  // triggers a single re-render at most.
  const handleEditorChange = useCallback((value: string) => {
    draftRef.current = value
    const next = value !== baselineRef.current
    setDirty(prev => (prev === next ? prev : next))
  }, [])

  // Publish the unsaved state to the rail so the tab can show a modified dot.
  // Keyed by url; cleared on unmount/tab-change so a stale dot never lingers.
  useEffect(() => {
    setPreviewDirty(target.url, editing && dirty)

    return () => setPreviewDirty(target.url, false)
  }, [target.url, editing, dirty])

  const beginEdit = () => {
    const text = state.text ?? ''
    baselineRef.current = text
    draftRef.current = text
    setDirty(false)
    setEditorKey(key => key + 1)
    setSaving(false)
    setSaveError(null)
    setConflict(false)
    setEditing(true)
  }

  // Latest `beginEdit` for the keydown listener, so the listener can stay
  // subscribed across renders without recreating itself or going stale.
  const beginEditRef = useRef(beginEdit)
  beginEditRef.current = beginEdit

  // Bare `e` enters edit mode when the file pane is hovered or focused and no
  // typable field has focus — a fast, button-free path (double-click felt laggy
  // because of the browser's click-disambiguation delay).
  useEffect(() => {
    if (!canEdit || editing) {
      return
    }

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'e' || event.metaKey || event.ctrlKey || event.altKey) {
        return
      }

      if (isTypableElement(document.activeElement)) {
        return
      }

      const root = readViewRef.current
      const focusWithin = Boolean(root && document.activeElement && root.contains(document.activeElement))

      if (!hoverRef.current && !focusWithin) {
        return
      }

      event.preventDefault()
      beginEditRef.current()
    }

    window.addEventListener('keydown', onKeyDown)

    return () => window.removeEventListener('keydown', onKeyDown)
  }, [canEdit, editing])

  const cancelEdit = () => {
    setEditing(false)
    setSaveError(null)
    setConflict(false)
  }

  const discardAndReload = () => {
    setEditing(false)
    setConflict(false)
    setSaveError(null)
    setSelfReload(n => n + 1)
  }

  const saveEdit = async (force = false) => {
    if (saving) {
      return
    }

    setSaving(true)
    setSaveError(null)

    try {
      // Stale-on-disk guard: re-read what's on disk now and compare to the
      // snapshot the user started from. If something changed underneath (an
      // agent edit, an external save), don't clobber it silently — surface the
      // choice. `force` is the user picking "overwrite" from that banner.
      if (!force) {
        try {
          const current = await readTextPreview(filePath)

          if (!current.binary && (current.text ?? '') !== baselineRef.current) {
            setConflict(true)
            setSaving(false)

            return
          }
        } catch {
          // Couldn't re-read for the check — fall through and attempt the write.
        }
      }

      await writeDesktopFileText(filePath, draftRef.current)
      baselineRef.current = draftRef.current
      setDirty(false)
      setConflict(false)
      setEditing(false)
      notifyWorkspaceChanged()
      setSelfReload(n => n + 1)
    } catch (error) {
      setSaveError(error instanceof Error ? error.message : String(error))
    } finally {
      setSaving(false)
    }
  }

  // Rendered before the loading/error branches so a background re-read (file
  // watcher, workspace tick) can't unmount the editor and drop the draft. Uses
  // the SAME container + fixed-height header as the read view so entering edit
  // never shifts the body — only the trailing controls and the body swap.
  if (editing) {
    return (
      <div className="flex h-full flex-col overflow-hidden bg-transparent">
        <PreviewModeSwitcher
          active="source"
          modes={[]}
          onSelect={() => {}}
          trailing={<EditControls dirty={dirty} onCancel={cancelEdit} onSave={() => void saveEdit()} saving={saving} />}
        />
        {conflict && (
          <div className="shrink-0 border-b border-amber-400/40 bg-amber-50 px-3 py-2 text-[0.7rem] text-amber-900 dark:border-amber-300/30 dark:bg-amber-300/10 dark:text-amber-100">
            <div className="font-semibold">{t.preview.diskChangedTitle}</div>
            <div className="mt-0.5 leading-relaxed">{t.preview.diskChangedBody}</div>
            <div className="mt-1.5 flex gap-3">
              <button
                className="font-bold underline underline-offset-4 transition-opacity hover:opacity-80"
                onClick={() => void saveEdit(true)}
                type="button"
              >
                {t.preview.overwrite}
              </button>
              <button
                className="font-bold underline underline-offset-4 transition-opacity hover:opacity-80"
                onClick={discardAndReload}
                type="button"
              >
                {t.preview.discardReload}
              </button>
            </div>
          </div>
        )}
        {saveError && (
          <div className="shrink-0 border-b border-destructive/40 bg-destructive/10 px-3 py-1.5 text-[0.7rem] text-destructive">
            {t.preview.saveFailed(saveError)}
          </div>
        )}
        <div className="min-h-0 flex-1 overflow-hidden">
          <CodeEditor
            filePath={filePath}
            initialValue={baselineRef.current}
            key={editorKey}
            onCancel={cancelEdit}
            onChange={handleEditorChange}
            onSave={() => void saveEdit()}
          />
        </div>
      </div>
    )
  }

  if (state.loading && state.text === undefined && state.dataUrl === undefined) {
    return <PageLoader label={t.preview.loading} />
  }

  if (state.error) {
    return <PreviewEmptyState body={state.error} title={t.preview.unavailable} />
  }

  if (pdfError) {
    return <PreviewEmptyState body={pdfError} title={t.preview.unavailable} />
  }

  if (
    !isImage &&
    !isPdf &&
    !forcePreview &&
    (target.binary || target.large || state.binary || (state.byteSize ?? 0) > TEXT_PREVIEW_MAX_BYTES)
  ) {
    const binary = target.binary || state.binary
    const size = target.byteSize || state.byteSize

    return (
      <PreviewEmptyState
        body={binary ? t.preview.binaryBody(target.label) : t.preview.largeBody(target.label, formatBytes(size))}
        primaryAction={{ label: t.preview.previewAnyway, onClick: () => setForcePreview(true) }}
        title={binary ? t.preview.binaryTitle : t.preview.largeTitle}
        tone="warning"
      />
    )
  }

  if (isImage && state.dataUrl) {
    return (
      <div className="flex h-full w-full items-center justify-center overflow-auto bg-transparent p-4">
        <img
          alt={target.label}
          className="max-h-full max-w-full rounded-lg object-contain shadow-sm"
          draggable={false}
          src={state.dataUrl}
        />
      </div>
    )
  }

  if (isPdf && state.dataUrl && pdfUrl) {
    return (
      <div className="h-full w-full overflow-hidden bg-transparent">
        <iframe
          aria-label={target.label}
          className="h-full w-full border-0 bg-white"
          src={pdfUrl}
          title={target.label}
        />
      </div>
    )
  }

  if (isPdf && state.dataUrl) {
    return <PageLoader label={t.preview.loading} />
  }

  if (isText && state.text !== undefined) {
    const isMarkdown = (state.language || target.language) === 'markdown'
    const hasDiff = Boolean(state.diff && state.diff.trim())
    // Order the toggle reads left→right; default lands on the most useful view.
    const modes: PreviewViewMode[] = []

    if (isMarkdown) {
      modes.push('rendered')
    }

    modes.push('source')

    if (hasDiff) {
      modes.push('diff')
    }

    const autoMode: PreviewViewMode = hasDiff ? 'diff' : isMarkdown ? 'rendered' : 'source'
    const mode = userMode && modes.includes(userMode) ? userMode : autoMode
    modeRef.current = mode

    return (
      <div
        className="flex h-full flex-col overflow-hidden bg-transparent"
        onMouseEnter={() => {
          hoverRef.current = true
        }}
        onMouseLeave={() => {
          hoverRef.current = false
        }}
        ref={readViewRef}
      >
        {state.truncated && (
          <div className="border-b border-border/60 bg-muted/35 px-3 py-1.5 text-[0.68rem] text-muted-foreground">
            {t.preview.truncated}
          </div>
        )}
        <PreviewModeSwitcher
          active={mode}
          modes={modes}
          onSelect={next => {
            // Capture where the user is (from the view being left) before
            // switching, so the new view can restore the same position. Diff
            // has no meaningful cross-view position - skip the capture.
            if (next !== mode && state.text) {
              if (mode === 'rendered') {
                pendingViewAnchorRef.current = capturePreviewAnchor()
              } else if (mode === 'source') {
                pendingViewAnchorRef.current = captureSourceAnchor()
              } else {
                pendingViewAnchorRef.current = null
              }
            }

            setUserMode(next)
          }}
          trailing={
            <>
              {jumpTarget && (
                <Tip label={t.preview.jumpToChangeTitle}>
                  <button
                    className="flex items-center gap-1 text-[0.625rem] font-bold text-muted-foreground underline-offset-4 transition-colors hover:text-foreground"
                    onClick={() => {
                      // Cycle through the changes: click 1 -> change 1, click
                      // 2 -> change 2, …; after the last change the next click
                      // restores where the user was when the change arrived,
                      // then the cycle restarts. In the rendered preview the
                      // change's anchor text is matched against a rendered
                      // block (fractional fallback); in the source view it
                      // scrolls line-exact.
                      const target = jumpTarget
                      const restore = target.index >= target.changes.length
                      const change = restore ? null : target.changes[target.index]

                      if (restore) {
                        // Back to the original reading position.
                        if (modeRef.current === 'rendered') {
                          const scroller = scrollerRef.current

                          if (scroller) {
                            if (target.original?.prefix) {
                              const block = previewBlockForPrefix(scroller, target.original.prefix)

                              if (block) {
                                scrollPreviewBlockIntoView(scroller, block)
                              } else {
                                const max = Math.max(0, scroller.scrollHeight - scroller.clientHeight)
                                scroller.scrollTop = target.original.fraction * max
                              }
                            } else {
                              scroller.scrollTop = target.originalScrollTop
                            }
                          }
                        } else if (modeRef.current === 'source' && target.original) {
                          const line = lineForCrossViewPrefix((state.text ?? '').split('\n'), target.original.prefix ?? '')

                          if (line) {
                            setScrollRequest({ line, nonce: Date.now() })
                          }
                        }
                      } else if (change) {
                        if (modeRef.current === 'rendered') {
                          const scroller = scrollerRef.current

                          if (!scroller) {return}

                          const block = change.anchor.prefix ? previewBlockForPrefix(scroller, change.anchor.prefix) : null

                          if (block) {
                            scrollPreviewBlockIntoView(scroller, block)
                          } else {
                            const max = Math.max(0, scroller.scrollHeight - scroller.clientHeight)
                            scroller.scrollTop = change.anchor.fraction * max
                          }
                        } else if (modeRef.current === 'source') {
                          setScrollRequest({ line: change.line, nonce: Date.now() })
                        } else {
                          // diff view: jump there means switching to source.
                          setUserMode('source')
                          setScrollRequest({ line: change.line, nonce: Date.now() })
                        }
                      }

                      setJumpTarget(prev =>
                        prev ? { ...prev, index: (prev.index + 1) % (prev.changes.length + 1), nonce: prev.nonce + 1 } : prev
                      )
                    }}
                    type="button"
                  >
                    ↓
                    <span>
                      {jumpTarget.index >= jumpTarget.changes.length
                        ? t.preview.backToReading
                        : `${t.preview.jumpToChange} ${jumpTarget.index + 1}/${jumpTarget.changes.length}`}
                    </span>
                  </button>
                </Tip>
              )}
              {canEdit ? (
                <Tip label={`${t.preview.edit} (e)`}>
                  <button
                    className="flex items-center gap-1 text-[0.625rem] font-bold text-muted-foreground underline-offset-4 transition-colors hover:text-foreground"
                    onClick={beginEdit}
                    type="button"
                  >
                    <Pencil className="size-3" />
                    {t.preview.edit}
                  </button>
                </Tip>
              ) : null}
            </>
          }
        />
        <div
          className="min-h-0 flex-1 overflow-auto"
          onCopy={
            mode === 'rendered'
              ? event => {
                  // Rebuild the copied text with LaTeX sources in place of
                  // Chromium's one-glyph-per-line MathML serialization.
                  const text = selectionTextWithMath(event.currentTarget)

                  if (text) {
                    event.clipboardData.setData('text/plain', text)
                    event.preventDefault()
                  }
                }
              : undefined
          }
          ref={scrollerRef}
        >
          {mode === 'rendered' ? (
            <MarkdownPreview text={state.text} />
          ) : mode === 'diff' ? (
            <FileDiffPanel
              className="mx-0 mb-0 h-full max-h-none"
              diff={state.diff ?? ''}
              fullText={state.text}
              path={filePath}
              showLineNumbers
            />
          ) : (
            <SourceView
              filePath={filePath}
              language={shikiLanguageForFilename(filePath) || state.language || 'text'}
              scrollNonce={scrollRequest?.nonce}
              scrollToLine={scrollRequest ? scrollRequest.line : undefined}
              text={state.text}
            />
          )}
        </div>
      </div>
    )
  }

  return <PreviewEmptyState body={t.preview.noInlineBody(target.mimeType || '')} title={t.preview.noInlineTitle} />
}

import type * as React from 'react'
import type {
  ComponentProps,
  CSSProperties,
  DragEvent as ReactDragEvent,
  MouseEvent as ReactMouseEvent,
  ReactNode
} from 'react'
import { useEffect, useMemo, useState } from 'react'
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

function PreviewToggle({ asSource, onToggle }: { asSource: boolean; onToggle: () => void }) {
  const { t } = useI18n()

  return (
    <div className="sticky top-0 z-10 flex justify-end border-b border-border/40 bg-transparent px-3 py-1 backdrop-blur">
      <button
        className="text-[0.625rem] font-bold text-muted-foreground underline decoration-current/20 underline-offset-4 transition-colors hover:text-foreground"
        onClick={onToggle}
        type="button"
      >
        {asSource ? t.preview.renderedPreview : t.preview.source}
      </button>
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

function SourceView({ filePath, language, text }: { filePath: string; language: string; text: string }) {
  const { t } = useI18n()
  const lineCount = useMemo(() => Math.max(1, text.split('\n').length), [text])
  const [selection, setSelection] = useState<LineSelection | null>(null)
  const inSelection = (line: number) => selection != null && line >= selection.start && line <= selection.end

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
    <div className="grid min-w-max grid-cols-[auto_minmax(0,1fr)] font-mono text-xs leading-relaxed">
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
        className="relative [&_pre]:m-0 [&_pre]:px-3 [&_pre]:py-3 [&_pre]:bg-transparent!"
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
        <ShikiHighlighter
          addDefaultStyles={false}
          as="div"
          defaultColor="light-dark()"
          delay={80}
          language={language || 'text'}
          showLanguage={false}
          theme={SHIKI_THEME}
        >
          {text}
        </ShikiHighlighter>
      </div>
    </div>
  )
}

export function LocalFilePreview({ reloadKey, target }: { reloadKey: number; target: PreviewTarget }) {
  const { t } = useI18n()
  const [state, setState] = useState<LocalPreviewState>({ loading: true })
  const [forcePreview, setForcePreview] = useState(false)
  const [renderMarkdownAsSource, setRenderMarkdownAsSource] = useState(false)
  const filePath = filePathForTarget(target)
  const isImage = target.previewKind === 'image'

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

  if (state.error) {
    return <PreviewEmptyState body={state.error} title={t.preview.unavailable} />
  }

  if (
    !isImage &&
    !forcePreview &&
    (target.binary || target.large || state.binary || (state.byteSize ?? 0) > TEXT_PREVIEW_MAX_BYTES)
  ) {
    const binary = target.binary || state.binary
    const size = target.byteSize || state.byteSize

    return (
      <PreviewEmptyState
        body={
          binary
            ? t.preview.binaryBody(target.label)
            : t.preview.largeBody(target.label, formatBytes(size))
        }
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

  if (isText && state.text !== undefined) {
    const isMarkdown = (state.language || target.language) === 'markdown'
    const showRendered = isMarkdown && !renderMarkdownAsSource

    return (
      <div className="h-full overflow-auto bg-transparent">
        {state.truncated && (
          <div className="border-b border-border/60 bg-muted/35 px-3 py-1.5 text-[0.68rem] text-muted-foreground">
            {t.preview.truncated}
          </div>
        )}
        {isMarkdown && <PreviewToggle asSource={!showRendered} onToggle={() => setRenderMarkdownAsSource(s => !s)} />}
        {showRendered ? (
          <MarkdownPreview text={state.text} />
        ) : (
          <SourceView filePath={filePath} language={state.language || 'text'} text={state.text} />
        )}
      </div>
    )
  }

  return (
    <PreviewEmptyState
      body={t.preview.noInlineBody(target.mimeType || '')}
      title={t.preview.noInlineTitle}
    />
  )
}

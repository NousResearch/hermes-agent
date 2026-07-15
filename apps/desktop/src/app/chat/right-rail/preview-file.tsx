import type * as React from 'react'
import type {
  ComponentProps,
  CSSProperties,
  DragEvent as ReactDragEvent,
  MouseEvent as ReactMouseEvent,
  ReactNode
} from 'react'
import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ShikiHighlighter from 'react-shiki'
import { Streamdown } from 'streamdown'

import { requestComposerFocus, requestComposerInsertRefs } from '@/app/chat/composer/focus'
import { droppedFileInlineRef } from '@/app/chat/composer/inline-refs'
import { HERMES_PATHS_MIME } from '@/app/chat/hooks/use-composer-actions'
import { isAddSelectionShortcut } from '@/app/right-sidebar/terminal/selection'
import { RichCodeBlock } from '@/components/assistant-ui/embeds'
import { CodeEditor } from '@/components/chat/code-editor'
import { FileDiffPanel } from '@/components/chat/diff-lines'
import { chunkTextLines, useFixedRowWindow } from '@/components/chat/fixed-row-window'
import { PageLoader } from '@/components/page-loader'
import { translateNow, useI18n } from '@/i18n'
import {
  desktopFileDiff,
  desktopGitRoot,
  readDesktopFileDataUrl,
  readDesktopFileText,
  writeDesktopFileText
} from '@/lib/desktop-fs'
import { Check, Pencil, X } from '@/lib/icons'
import { shikiLanguageForFilename } from '@/lib/markdown-code'
import { cn } from '@/lib/utils'
import type { PreviewTarget } from '@/store/preview'
import { setPreviewDirty } from '@/store/preview-edit'
import { $currentCwd } from '@/store/session'
import { notifyWorkspaceChanged } from '@/store/workspace-events'

const SHIKI_THEME = { dark: 'github-dark-default', light: 'github-light-default' } as const
const TEXT_PREVIEW_MAX_BYTES = 512 * 1024
const SOURCE_CHUNK_LINES = 200
const SOURCE_LINE_PX = 20
const SOURCE_OVERSCAN_LINES = 400

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

function PreviewModeSwitcher({
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

function SourceView({ filePath, language, text }: { filePath: string; language: string; text: string }) {
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
      // Insert into and focus the SAME composer — 'active' — so a tile that owns
      // focus keeps it instead of the ref landing in a tile but main stealing focus.
      requestComposerInsertRefs([ref])
      requestComposerFocus('active')
    }

    window.addEventListener('keydown', onKeyDown, { capture: true })

    return () => window.removeEventListener('keydown', onKeyDown, { capture: true })
  }, [filePath, selection])

  return (
    <div className="h-full overflow-auto" onScroll={onScroll} ref={scrollerRef}>
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
                      'h-5 w-9 cursor-pointer pr-2 leading-5 tabular-nums transition-colors',
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

type PreviewViewMode = 'diff' | 'rendered' | 'source'

export function LocalFilePreview({ reloadKey, target }: { reloadKey: number; target: PreviewTarget }) {
  const { t } = useI18n()
  const [state, setState] = useState<LocalPreviewState>({ loading: true })
  const [forcePreview, setForcePreview] = useState(false)
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
  // For the bare-`e` shortcut: the read-view root (to detect focus-within) and a
  // hover flag (no state — only the keydown handler reads it).
  const readViewRef = useRef<HTMLDivElement>(null)
  const hoverRef = useRef(false)
  const filePath = filePathForTarget(target)
  const isImage = target.previewKind === 'image'

  useEffect(() => {
    setUserMode(null)
    setEditing(false)
    setDirty(false)
    setSaving(false)
    setSaveError(null)
    setConflict(false)
    draftRef.current = ''
    baselineRef.current = ''
  }, [filePath, reloadKey])

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
  }, [blockedByTarget, filePath, forcePreview, isImage, isText, reloadKey, selfReload, target.dataUrl, target.language])

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
          onSelect={setUserMode}
          trailing={
            canEdit ? (
              <button
                className="flex items-center gap-1 text-[0.625rem] font-bold text-muted-foreground underline-offset-4 transition-colors hover:text-foreground"
                onClick={beginEdit}
                title={`${t.preview.edit} (e)`}
                type="button"
              >
                <Pencil className="size-3" />
                {t.preview.edit}
              </button>
            ) : null
          }
        />
        <div className="min-h-0 flex-1 overflow-auto">
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
              text={state.text}
            />
          )}
        </div>
      </div>
    )
  }

  return <PreviewEmptyState body={t.preview.noInlineBody(target.mimeType || '')} title={t.preview.noInlineTitle} />
}

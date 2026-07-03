import { defaultKeymap, history, historyKeymap, indentWithTab } from '@codemirror/commands'
import { bracketMatching, indentOnInput, LanguageDescription } from '@codemirror/language'
import { languages } from '@codemirror/language-data'
import { Compartment, EditorState } from '@codemirror/state'
import { Decoration, drawSelection, EditorView, keymap, lineNumbers } from '@codemirror/view'
import { type RefObject, useEffect, useRef } from 'react'

import { tryFormatJson } from '@/lib/json-format'
import { cn } from '@/lib/utils'
import { useTheme } from '@/themes/context'

import { githubEditorTheme } from './code-editor-theme'

type FormatOutcome = { ok: true } | { ok: false; error: string }

function applyFormatJson(view: EditorView, onError?: (error: string) => void): FormatOutcome {
  const text = view.state.doc.toString()
  const result = tryFormatJson(text)

  if (!result.ok) {
    onError?.(result.error)

    return result
  }

  if (result.text !== text) {
    view.dispatch({ changes: { from: 0, insert: result.text, to: view.state.doc.length } })
  }

  return { ok: true }
}

/** Imperative surface for callers that drive selection from outside (e.g. a
 *  config list focusing its block in the document). */
export interface CodeEditorApi {
  formatJson: () => FormatOutcome
  setCursor: (pos: number) => void
}

interface CodeEditorProps {
  apiRef?: RefObject<CodeEditorApi | null>
  className?: string
  /** Read-only: block edits (e.g. while a save is in flight) without unmounting. */
  disabled?: boolean
  /** Mod-Shift-F + `apiRef.formatJson()`. In-memory JSON docs only. */
  formatJson?: boolean
  /**
   * Standalone chrome: rounded border on an outer shell. The CodeMirror surface
   * inside is identical to pane previews (no extra inset). Off by default.
   */
  framed?: boolean
  filePath: string
  /** Character range to wash with a subtle background (the "you are here" block). */
  highlight?: null | { from: number; to: number }
  // Read once at mount. To load a different file or discard edits, remount the
  // component (give it a new React `key`) rather than pushing a new value in.
  initialValue: string
  onCancel?: () => void
  onChange: (value: string) => void
  /** Button or Mod-Shift-F. */
  onFormatJsonError?: (error: string) => void
  /** Fires with the primary cursor offset whenever the selection moves. */
  onCursorChange?: (pos: number) => void
  onSave?: () => void
}

// Focus treatment for the active range: a subtle wash on its lines, and
// everything OUTSIDE dimmed — the document recedes so the block you're in
// reads as "you are here".
function blockHighlight(range: { from: number; to: number }) {
  return EditorView.decorations.compute([], state => {
    const clamp = (pos: number) => Math.max(0, Math.min(pos, state.doc.length))
    const active = Decoration.line({ class: 'cm-hermes-active-block' })
    // Inline style, not a theme class: theme rules are scoped per-extension
    // and line opacity must never lose that fight.
    const dimmed = Decoration.line({ attributes: { style: 'opacity:0.5;transition:opacity 120ms ease-out' } })
    const first = state.doc.lineAt(clamp(range.from)).number
    const last = state.doc.lineAt(clamp(range.to)).number
    const marks = []

    for (let n = 1; n <= state.doc.lines; n++) {
      marks.push((n >= first && n <= last ? active : dimmed).range(state.doc.line(n).from))
    }

    return Decoration.set(marks)
  })
}

function baseName(filePath: string): string {
  const cleaned = filePath.replace(/[\\/]+$/, '')

  return (
    cleaned
      .slice(cleaned.lastIndexOf('/') + 1)
      .split('\\')
      .pop() ?? cleaned
  )
}

// Mirror SourceView's geometry/typography 1:1 so toggling preview⇄edit never
// shifts the file. CM's base stylesheet targets some of these with two-class
// selectors (e.g. `.cm-lineNumbers .cm-gutterElement`) that out-specify a bare
// `.cm-gutterElement` rule, so we match that specificity to win. SourceView
// reference: font var(--font-mono)/0.7rem/400, 1.25rem rows, gutter w-9 + pr-2
// (muted/55), code 0.625rem line inset.
const MONO_FONT = 'var(--font-mono)'
const ROW_HEIGHT = '1.25rem'
const CODE_SIZE = '0.7rem'
const GUTTER_COLOR = 'color-mix(in oklab, var(--muted-foreground) 55%, transparent)'

const LAYOUT_THEME = EditorView.theme({
  '&': {
    WebkitFontSmoothing: 'antialiased',
    backgroundColor: 'transparent',
    height: '100%'
  },
  // CM's base theme ships `.cm-content { padding: 4px 0 }` (~5px top/bottom).
  // Zero it explicitly so pane + framed interiors match SourceView flush-top.
  '.cm-content': {
    fontFamily: MONO_FONT,
    fontSize: CODE_SIZE,
    fontWeight: '400',
    lineHeight: ROW_HEIGHT,
    padding: '0',
    paddingBottom: '0',
    paddingTop: '0'
  },
  '.cm-gutters': {
    backgroundColor: 'transparent',
    border: 'none',
    color: GUTTER_COLOR,
    fontFamily: MONO_FONT,
    fontSize: CODE_SIZE
  },
  // Two-class selector to beat CM's base `.cm-lineNumbers .cm-gutterElement`.
  '.cm-lineNumbers .cm-gutterElement': {
    boxSizing: 'border-box',
    fontVariantNumeric: 'tabular-nums',
    fontWeight: '400',
    lineHeight: ROW_HEIGHT,
    minWidth: '2.25rem',
    padding: '0 0.5rem 0 0',
    textAlign: 'right'
  },
  '.cm-line': {
    fontFamily: MONO_FONT,
    fontSize: CODE_SIZE,
    fontWeight: '400',
    lineHeight: ROW_HEIGHT,
    padding: '0 0.625rem'
  },
  '.cm-scroller': {
    fontFamily: MONO_FONT,
    fontSize: CODE_SIZE,
    lineHeight: ROW_HEIGHT,
    overflow: 'auto'
  },
  '.cm-hermes-active-block': {
    backgroundColor: 'color-mix(in srgb, var(--dt-foreground) 5%, transparent)'
  }
})

// Framed = prose editing (SOUL.md, skills, memories): no line-number gutter (it
// shoved text right and made the left inset dwarf the top), and zero the line's
// own horizontal padding so the host's uniform `p-2` is the ONLY inset — even
// breathing room on all four sides. Long lines wrap rather than scroll.
const FRAMED_THEME = EditorView.theme({
  '.cm-line': { padding: '0' }
})

// A deliberately small CodeMirror 6 surface for *spot edits* — not an IDE: line
// numbers, history, selection, bracket matching, syntax highlighting. No fold
// gutter, autocomplete, or active-line chrome, so it reads like the preview it
// replaces. It owns its own buffer; the parent tracks dirty via `onChange` and
// resets by remounting. ⌘/Ctrl+S and ⌘/Ctrl+Enter save; Esc cancels; the app's
// light/dark mode is followed live without losing the cursor.
export function CodeEditor({
  apiRef,
  className,
  disabled = false,
  formatJson = false,
  framed = false,
  filePath,
  highlight,
  initialValue,
  onCancel,
  onChange,
  onCursorChange,
  onFormatJsonError,
  onSave
}: CodeEditorProps) {
  const { resolvedMode } = useTheme()
  const hostRef = useRef<HTMLDivElement | null>(null)
  const viewRef = useRef<EditorView | null>(null)
  const languageConf = useRef(new Compartment())
  const themeConf = useRef(new Compartment())
  const highlightConf = useRef(new Compartment())
  const editableConf = useRef(new Compartment())
  const onCancelRef = useRef(onCancel)
  const onChangeRef = useRef(onChange)
  const onCursorChangeRef = useRef(onCursorChange)
  const onFormatJsonErrorRef = useRef(onFormatJsonError)
  const onSaveRef = useRef(onSave)
  const formatJsonRef = useRef(formatJson)
  onCancelRef.current = onCancel
  onChangeRef.current = onChange
  onCursorChangeRef.current = onCursorChange
  onFormatJsonErrorRef.current = onFormatJsonError
  onSaveRef.current = onSave
  formatJsonRef.current = formatJson

  useEffect(() => {
    const host = hostRef.current

    if (!host) {
      return
    }

    const isDark = resolvedMode === 'dark'

    const save = () => {
      onSaveRef.current?.()

      return true
    }

    const runFormatJson = () => {
      if (!formatJsonRef.current || !viewRef.current) {
        return false
      }

      applyFormatJson(viewRef.current, error => onFormatJsonErrorRef.current?.(error))

      return true
    }

    const state = EditorState.create({
      doc: initialValue,
      extensions: [
        // Gutter only outside framed mode — framed prose reads better flush.
        ...(framed ? [] : [lineNumbers()]),
        history(),
        drawSelection(),
        indentOnInput(),
        bracketMatching(),
        keymap.of([
          ...defaultKeymap,
          ...historyKeymap,
          indentWithTab,
          { key: 'Mod-s', preventDefault: true, run: save },
          { key: 'Mod-Enter', preventDefault: true, run: save },
          ...(formatJson ? [{ key: 'Mod-Shift-f', preventDefault: true, run: runFormatJson }] : []),
          {
            key: 'Escape',
            run: () => {
              if (!onCancelRef.current) {
                return false
              }

              onCancelRef.current()

              return true
            }
          }
        ]),
        languageConf.current.of([]),
        themeConf.current.of(githubEditorTheme(isDark)),
        highlightConf.current.of([]),
        editableConf.current.of(EditorState.readOnly.of(disabled)),
        EditorView.updateListener.of(update => {
          if (update.docChanged) {
            onChangeRef.current(update.state.doc.toString())
          }

          if (update.selectionSet || update.docChanged) {
            onCursorChangeRef.current?.(update.state.selection.main.head)
          }
        }),
        LAYOUT_THEME,
        // Standalone edits (SOUL.md, skills, memories) are prose, not code —
        // wrap long lines instead of scrolling horizontally, and drop the gutter
        // inset. Pane previews stay flush/scrolling to mirror their SourceView.
        ...(framed ? [EditorView.lineWrapping, FRAMED_THEME] : [])
      ]
    })

    const view = new EditorView({ parent: host, state })
    viewRef.current = view

    if (apiRef) {
      apiRef.current = {
        formatJson: () => {
          const view = viewRef.current

          if (!view || !formatJsonRef.current) {
            return { ok: false, error: 'JSON formatting is not enabled for this editor' }
          }

          return applyFormatJson(view)
        },
        setCursor: pos => {
          const clamped = Math.max(0, Math.min(pos, view.state.doc.length))
          view.dispatch({ scrollIntoView: true, selection: { anchor: clamped } })
          view.focus()
        }
      }
    }

    // Focus on mount so entering edit mode (button or double-click) lands the
    // caret in the buffer ready to type, no extra click required.
    view.focus()

    return () => {
      view.destroy()
      viewRef.current = null

      if (apiRef) {
        apiRef.current = null
      }
    }
    // Created once per mount; the parent remounts (via `key`) to load a new
    // file or discard. Theme/language are applied reactively below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Load + apply syntax highlighting for the file's language (lazy per language).
  useEffect(() => {
    let cancelled = false
    const description = LanguageDescription.matchFilename(languages, baseName(filePath))

    if (!description) {
      viewRef.current?.dispatch({ effects: languageConf.current.reconfigure([]) })

      return
    }

    void description.load().then(support => {
      if (!cancelled && viewRef.current) {
        viewRef.current.dispatch({ effects: languageConf.current.reconfigure(support) })
      }
    })

    return () => {
      cancelled = true
    }
  }, [filePath])

  useEffect(() => {
    viewRef.current?.dispatch({
      effects: themeConf.current.reconfigure(githubEditorTheme(resolvedMode === 'dark'))
    })
  }, [resolvedMode])

  const highlightFrom = highlight?.from
  const highlightTo = highlight?.to

  useEffect(() => {
    viewRef.current?.dispatch({
      effects: highlightConf.current.reconfigure(
        highlightFrom !== undefined && highlightTo !== undefined
          ? blockHighlight({ from: highlightFrom, to: highlightTo })
          : []
      )
    })
  }, [highlightFrom, highlightTo])

  useEffect(() => {
    viewRef.current?.dispatch({ effects: editableConf.current.reconfigure(EditorState.readOnly.of(disabled)) })
  }, [disabled])

  if (!framed) {
    return <div className={cn('h-full min-h-0 overflow-hidden', className)} ref={hostRef} />
  }

  // Border on the shell only — inner body matches preview-file / DetailPane:
  // <div className="min-h-0 flex-1 overflow-hidden"><CodeEditor /></div>
  return (
    <div
      className={cn(
        'flex h-full min-h-0 flex-col overflow-hidden rounded-md border border-(--ui-stroke-tertiary)',
        className
      )}
    >
      {/* Padding lives on the CM *mount node* itself — outside CodeMirror's
          DOM entirely, so its `.cm-content { padding: 0 }` can't fight it. This
          is why every prior attempt (Tailwind on .cm-content, scroller padding)
          lost: they targeted CM-owned nodes. This div isn't one. */}
      <div className="min-h-0 flex-1 overflow-hidden p-2" ref={hostRef} />
    </div>
  )
}

import { defaultKeymap, history, historyKeymap, indentWithTab } from '@codemirror/commands'
import { bracketMatching, indentOnInput, LanguageDescription } from '@codemirror/language'
import { languages } from '@codemirror/language-data'
import {
  getSearchQuery,
  openSearchPanel,
  search,
  searchKeymap,
  searchPanelOpen,
  type SearchQuery
} from '@codemirror/search'
import { Compartment, EditorState } from '@codemirror/state'
import { Decoration, drawSelection, EditorView, keymap, lineNumbers } from '@codemirror/view'
import { type RefObject, useEffect, useRef } from 'react'

import { useI18n } from '@/i18n'
import type { Translations } from '@/i18n/types'
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
  findReplace: () => boolean
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

type EditorSearchCopy = Translations['editorSearch']

function searchPhrases(copy: EditorSearchCopy): Record<string, string> {
  return {
    Find: copy.find,
    Replace: copy.replace,
    all: copy.selectAll,
    'by word': copy.wholeWord,
    close: copy.close,
    'current match': copy.currentMatch,
    'match case': copy.matchCase,
    next: copy.next,
    'on line': copy.onLine,
    previous: copy.previous,
    regexp: copy.regexp,
    replace: copy.replace,
    'replace all': copy.replaceAll,
    'replaced $ matches': copy.replacedMatches,
    'replaced match on line $': copy.replacedMatchOnLine
  }
}

function searchTargetChanged(previous: SearchQuery, next: SearchQuery): boolean {
  return (
    previous.search !== next.search ||
    previous.caseSensitive !== next.caseSensitive ||
    previous.regexp !== next.regexp ||
    previous.wholeWord !== next.wholeWord
  )
}

function selectFirstSearchMatch(view: EditorView, query: SearchQuery) {
  if (!query.valid || !query.search) {
    return
  }

  const first = query.getCursor(view.state).next()

  if (!first.done) {
    view.dispatch({
      scrollIntoView: true,
      selection: { anchor: first.value.from, head: first.value.to }
    })
  }
}

function updateSearchMatchStatus(view: EditorView, copy: EditorSearchCopy) {
  const panel = view.dom.querySelector<HTMLElement>('.cm-search')

  if (!panel) {
    return
  }

  let status = panel.querySelector<HTMLElement>('.cm-search-match-status')

  if (!status) {
    status = document.createElement('span')
    status.className = 'cm-search-match-status'
    status.setAttribute('aria-live', 'polite')
    status.setAttribute('role', 'status')
    panel.insertBefore(status, panel.querySelector('button[name="close"]'))
  }

  const query = getSearchQuery(view.state)

  if (!query.search) {
    status.textContent = ''

    return
  }

  if (!query.valid) {
    status.textContent = copy.invalidRegex

    return
  }

  const selection = view.state.selection.main
  const cursor = query.getCursor(view.state)
  let current = -1
  let nextAtOrAfterCursor = -1
  let total = 0

  for (let next = cursor.next(); !next.done; next = cursor.next()) {
    const match = next.value

    if (current < 0 && match.from === selection.from && match.to === selection.to) {
      current = total
    }

    if (nextAtOrAfterCursor < 0 && match.from >= selection.head) {
      nextAtOrAfterCursor = total
    }

    total++
  }

  if (total === 0) {
    status.textContent = copy.noResults

    return
  }

  if (current < 0) {
    current = nextAtOrAfterCursor
  }

  status.textContent = copy.matchCount((current < 0 ? 0 : current) + 1, total)
}

function refreshSearchMatchStatus(view: EditorView, copy: EditorSearchCopy) {
  if (view.dom.querySelector('.cm-search')) {
    updateSearchMatchStatus(view, copy)

    return
  }

  queueMicrotask(() => {
    if (view.dom.isConnected) {
      updateSearchMatchStatus(view, copy)
    }
  })
}

function openFindPanel(view: EditorView, copy: EditorSearchCopy): boolean {
  openSearchPanel(view)
  refreshSearchMatchStatus(view, copy)

  return true
}

function openFindReplacePanel(view: EditorView, copy: EditorSearchCopy): boolean {
  openFindPanel(view, copy)

  queueMicrotask(() => {
    const field = view.dom.querySelector<HTMLInputElement>('.cm-search input[name="replace"]')

    field?.focus()
    field?.select()
  })

  return true
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
  '.cm-search': {
    alignItems: 'center',
    backgroundColor: 'var(--ui-bg-elevated)',
    display: 'grid',
    fontFamily: 'var(--font-sans)',
    fontSize: '0.625rem',
    gap: '0.25rem 0.375rem',
    gridTemplateColumns: 'minmax(7rem, 1fr) auto auto auto',
    overflowX: 'auto',
    padding: '0.375rem 1.75rem 0.375rem 0.5rem'
  },
  '.cm-search br': {
    display: 'none'
  },
  '.cm-search .cm-textfield': {
    backgroundColor: 'var(--dt-background)',
    border: '1px solid var(--dt-input)',
    borderRadius: '0.25rem',
    color: 'var(--dt-foreground)',
    fontFamily: 'var(--font-sans)',
    fontSize: '0.625rem',
    minWidth: '0',
    padding: '0.1875rem 0.375rem',
    width: '100%'
  },
  '.cm-search .cm-textfield::placeholder': {
    color: 'color-mix(in srgb, var(--dt-foreground) 58%, transparent)'
  },
  '.cm-search .cm-button': {
    backgroundColor: 'var(--dt-secondary)',
    backgroundImage: 'none',
    border: '1px solid var(--dt-border)',
    borderRadius: '0.25rem',
    color: 'var(--dt-secondary-foreground)',
    fontFamily: 'var(--font-sans)',
    fontSize: '0.625rem',
    lineHeight: '1rem',
    padding: '0.125rem 0.375rem',
    whiteSpace: 'nowrap'
  },
  '.cm-search label': {
    alignItems: 'center',
    color: 'color-mix(in srgb, var(--dt-foreground) 72%, transparent)',
    display: 'inline-flex',
    gap: '0.1875rem',
    whiteSpace: 'nowrap'
  },
  '.cm-search input[type="checkbox"]': {
    accentColor: 'var(--dt-primary)',
    height: '0.75rem',
    margin: '0',
    width: '0.75rem'
  },
  '.cm-search input[name="search"]': { gridColumn: '1', gridRow: '1' },
  '.cm-search button[name="next"]': { gridColumn: '2', gridRow: '1' },
  '.cm-search button[name="prev"]': { gridColumn: '3', gridRow: '1' },
  '.cm-search button[name="select"]': { gridColumn: '4', gridRow: '1' },
  '.cm-search label:has(input[name="case"])': { gridColumn: '1', gridRow: '2' },
  '.cm-search label:has(input[name="re"])': { gridColumn: '2', gridRow: '2' },
  '.cm-search label:has(input[name="word"])': { gridColumn: '3', gridRow: '2' },
  '.cm-search input[name="replace"]': { gridColumn: '1', gridRow: '3' },
  '.cm-search button[name="replace"]': { gridColumn: '2', gridRow: '3' },
  '.cm-search button[name="replaceAll"]': { gridColumn: '3', gridRow: '3' },
  '.cm-search-match-status': {
    color: 'color-mix(in srgb, var(--dt-foreground) 82%, transparent)',
    fontVariantNumeric: 'tabular-nums',
    gridColumn: '4',
    gridRow: '2',
    justifySelf: 'end',
    whiteSpace: 'nowrap'
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

// A deliberately compact CodeMirror 6 surface for *spot edits* — not an IDE:
// line numbers, history, selection, bracket matching, syntax highlighting, and
// a full Windows-style Find/Replace panel. No fold gutter, autocomplete, or
// active-line chrome, so it reads like the preview it replaces. It owns its own
// buffer; the parent tracks dirty via `onChange` and resets by remounting.
// ⌘/Ctrl+F finds, Ctrl+H (⌘⌥F on macOS) opens replacement, ⌘/Ctrl+S and
// ⌘/Ctrl+Enter save, and Esc closes search before it cancels editing.
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
  const { t } = useI18n()
  const { resolvedMode } = useTheme()
  const hostRef = useRef<HTMLDivElement | null>(null)
  const viewRef = useRef<EditorView | null>(null)
  const languageConf = useRef(new Compartment())
  const searchPhrasesConf = useRef(new Compartment())
  const themeConf = useRef(new Compartment())
  const highlightConf = useRef(new Compartment())
  const editableConf = useRef(new Compartment())
  const onCancelRef = useRef(onCancel)
  const onChangeRef = useRef(onChange)
  const onCursorChangeRef = useRef(onCursorChange)
  const onFormatJsonErrorRef = useRef(onFormatJsonError)
  const onSaveRef = useRef(onSave)
  const formatJsonRef = useRef(formatJson)
  const searchCopyRef = useRef(t.editorSearch)
  onCancelRef.current = onCancel
  onChangeRef.current = onChange
  onCursorChangeRef.current = onCursorChange
  onFormatJsonErrorRef.current = onFormatJsonError
  onSaveRef.current = onSave
  formatJsonRef.current = formatJson
  searchCopyRef.current = t.editorSearch

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

    const runFind = (view: EditorView) => openFindPanel(view, searchCopyRef.current)
    const runFindReplace = (view: EditorView) => openFindReplacePanel(view, searchCopyRef.current)

    const state = EditorState.create({
      doc: initialValue,
      extensions: [
        // Gutter only outside framed mode — framed prose reads better flush.
        ...(framed ? [] : [lineNumbers()]),
        history(),
        EditorState.allowMultipleSelections.of(true),
        drawSelection(),
        indentOnInput(),
        bracketMatching(),
        search(),
        keymap.of([
          { key: 'Mod-f', preventDefault: true, run: runFind },
          { key: 'Ctrl-h', mac: 'Mod-Alt-f', preventDefault: true, run: runFindReplace },
          ...searchKeymap,
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
        searchPhrasesConf.current.of(EditorState.phrases.of(searchPhrases(searchCopyRef.current))),
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

          if (searchPanelOpen(update.state)) {
            const previousQuery = getSearchQuery(update.startState)
            const nextQuery = getSearchQuery(update.state)

            if (searchTargetChanged(previousQuery, nextQuery)) {
              queueMicrotask(() => {
                const currentQuery = getSearchQuery(update.view.state)

                if (viewRef.current === update.view && !searchTargetChanged(nextQuery, currentQuery)) {
                  selectFirstSearchMatch(update.view, currentQuery)
                }
              })
            }

            refreshSearchMatchStatus(update.view, searchCopyRef.current)
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
        findReplace: () => {
          const view = viewRef.current

          return view ? openFindReplacePanel(view, searchCopyRef.current) : false
        },
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

  useEffect(() => {
    const view = viewRef.current

    view?.dispatch({
      effects: searchPhrasesConf.current.reconfigure(EditorState.phrases.of(searchPhrases(t.editorSearch)))
    })

    if (view && searchPanelOpen(view.state)) {
      refreshSearchMatchStatus(view, t.editorSearch)
    }
  }, [t.editorSearch])

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

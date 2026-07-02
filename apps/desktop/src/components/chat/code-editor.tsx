import { defaultKeymap, history, historyKeymap, indentWithTab } from '@codemirror/commands'
import { bracketMatching, indentOnInput, LanguageDescription } from '@codemirror/language'
import { languages } from '@codemirror/language-data'
import { Compartment, EditorState } from '@codemirror/state'
import { drawSelection, EditorView, keymap, lineNumbers } from '@codemirror/view'
import { useEffect, useRef } from 'react'

import { cn } from '@/lib/utils'
import { useTheme } from '@/themes/context'

import { githubEditorTheme } from './code-editor-theme'

interface CodeEditorProps {
  className?: string
  filePath: string
  // Read once at mount. To load a different file or discard edits, remount the
  // component (give it a new React `key`) rather than pushing a new value in.
  initialValue: string
  onCancel?: () => void
  onChange: (value: string) => void
  onSave?: () => void
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
  '.cm-content': {
    fontFamily: MONO_FONT,
    fontSize: CODE_SIZE,
    fontWeight: '400',
    lineHeight: ROW_HEIGHT,
    padding: '0'
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
  }
})

// A deliberately small CodeMirror 6 surface for *spot edits* — not an IDE: line
// numbers, history, selection, bracket matching, syntax highlighting. No fold
// gutter, autocomplete, or active-line chrome, so it reads like the preview it
// replaces. It owns its own buffer; the parent tracks dirty via `onChange` and
// resets by remounting. ⌘/Ctrl+S and ⌘/Ctrl+Enter save; Esc cancels; the app's
// light/dark mode is followed live without losing the cursor.
export function CodeEditor({ className, filePath, initialValue, onCancel, onChange, onSave }: CodeEditorProps) {
  const { resolvedMode } = useTheme()
  const hostRef = useRef<HTMLDivElement | null>(null)
  const viewRef = useRef<EditorView | null>(null)
  const languageConf = useRef(new Compartment())
  const themeConf = useRef(new Compartment())
  const onCancelRef = useRef(onCancel)
  const onChangeRef = useRef(onChange)
  const onSaveRef = useRef(onSave)
  onCancelRef.current = onCancel
  onChangeRef.current = onChange
  onSaveRef.current = onSave

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

    const state = EditorState.create({
      doc: initialValue,
      extensions: [
        lineNumbers(),
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
        EditorView.updateListener.of(update => {
          if (update.docChanged) {
            onChangeRef.current(update.state.doc.toString())
          }
        }),
        LAYOUT_THEME
      ]
    })

    const view = new EditorView({ parent: host, state })
    viewRef.current = view
    // Focus on mount so entering edit mode (button or double-click) lands the
    // caret in the buffer ready to type, no extra click required.
    view.focus()

    return () => {
      view.destroy()
      viewRef.current = null
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

  return <div className={cn('h-full min-h-0 overflow-hidden', className)} ref={hostRef} />
}

import { defaultKeymap, history, historyKeymap } from '@codemirror/commands'
import { EditorState, type Extension } from '@codemirror/state'
import { drawSelection, EditorView, keymap } from '@codemirror/view'
import { useEffect, useRef } from 'react'

import { cn } from '@/lib/utils'

// Prose in the app's own face: no gutter, no language, no bracket matching.
// `components/chat/code-editor` is the CODE surface (mono, line numbers,
// syntax) — bending its chrome for a blank page would fight its contract, so
// this is its own small CodeMirror setup on the same primitives. It exists
// because the page must style parts of its text differently (an agent's reply
// against the user's own writing), which a plain textarea cannot do.
const PAGE_THEME = EditorView.theme({
  '&': {
    backgroundColor: 'transparent',
    color: 'var(--ui-text-primary)',
    height: '100%'
  },
  '&.cm-focused': { outline: 'none' },
  '.cm-content': {
    fontFamily: 'var(--dt-font-sans)',
    fontSize: '0.9375rem',
    lineHeight: '1.65',
    padding: '0 0 2rem'
  },
  '.cm-line': { padding: '0' },
  // The column is centered with the SCROLLER's padding, so the scrollbar stays
  // at the window edge and a click in the margin still lands in the text.
  '.cm-scroller': {
    fontFamily: 'var(--dt-font-sans)',
    lineHeight: '1.65',
    overflowY: 'auto',
    paddingInline: 'max(2rem, calc((100% - 48rem) / 2))'
  },
  // `drawSelection` paints its own caret and selection, and CodeMirror's base
  // theme colours them for a light page (a black caret, invisible here). Both
  // take app tokens instead, so they follow light and dark with everything
  // else. `.cm-focused` is repeated to match the base rule's specificity —
  // ours is declared later, so equal specificity wins.
  '.cm-cursor, .cm-dropCursor': { borderLeftColor: 'var(--ui-text-primary)' },
  '&.cm-focused.cm-focused > .cm-scroller > .cm-selectionLayer .cm-selectionBackground, .cm-selectionBackground': {
    backgroundColor: 'var(--ui-selection-background)'
  },
  '.cm-content ::selection': { backgroundColor: 'var(--ui-selection-background)' }
})

interface BackworkspaceEditorProps {
  ariaLabel: string
  autoFocus?: boolean
  className?: string
  /** Read once at mount, like the document (the `@` list lives here). */
  extensions?: Extension
  // Read once at mount, like `CodeEditor`: the view owns the document from
  // then on. To show another page, remount with a new React `key`.
  initialValue: string
  onChange: (value: string) => void
}

export function BackworkspaceEditor({
  ariaLabel,
  autoFocus = false,
  className,
  extensions,
  initialValue,
  onChange
}: BackworkspaceEditorProps) {
  const hostRef = useRef<HTMLDivElement | null>(null)
  const onChangeRef = useRef(onChange)

  onChangeRef.current = onChange

  useEffect(() => {
    const host = hostRef.current

    if (!host) {
      return
    }

    const view = new EditorView({
      parent: host,
      state: EditorState.create({
        doc: initialValue,
        extensions: [
          history(),
          drawSelection(),
          // Escape is left to the page: with an empty selection CodeMirror's
          // own binding reports it unhandled, so it reaches the section that
          // turns the window back.
          keymap.of([...defaultKeymap, ...historyKeymap]),
          EditorView.lineWrapping,
          EditorView.contentAttributes.of({ 'aria-label': ariaLabel, spellcheck: 'true' }),
          // A click in the centered column's margin lands on the scroller, not
          // on the text, so CodeMirror would ignore it. The page is meant to
          // take writing from anywhere on it: hand focus back instead.
          EditorView.domEventHandlers({
            mousedown(event, view) {
              if (event.target === view.scrollDOM) {
                view.focus()
              }

              return false
            }
          }),
          EditorView.updateListener.of(update => {
            if (update.docChanged) {
              onChangeRef.current(update.state.doc.toString())
            }
          }),
          PAGE_THEME,
          extensions ?? []
        ]
      })
    })

    if (autoFocus) {
      view.focus()
    }

    return () => view.destroy()
    // The document is mount-owned; re-running this would discard the user's edits.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return <div className={cn('min-h-0 flex-1 overflow-hidden', className)} ref={hostRef} />
}

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
  // The writing is a hand; the agent's replies are not, and quote-decorations
  // puts them back in the reading face. That is the whole distinction between
  // the two voices on the page, and it costs one declaration each.
  '.cm-content': {
    color: 'var(--bw-ink)',
    fontFamily: 'var(--bw-hand)',
    fontSize: '1.6875rem',
    lineHeight: '1.5',
    padding: '0 0 2rem'
  },
  '.cm-line': { padding: '0' },
  // The column is centered with the SCROLLER's padding, so the scrollbar stays
  // at the window edge. It has to be the scroller: the selection of several
  // lines is painted from the CONTENT's box edge inwards by whatever padding
  // `.cm-line` carries, so a margin held by either of those two would have the
  // highlight run the full width of the window. The margin is still writable —
  // `caretFromMargin` below hands a click there to the line beside it.
  // The gutter is reserved before there is anything to scroll, so the column
  // does not move when the page first outgrows the window. Without it a scroll
  // bar appears, the content box narrows by its width, and every line re-wraps —
  // which is how a command being read out loud broke across `rm -` / `rf`.
  // One side only, and the column sits half a scroll bar left of centre for it.
  // `both-edges` would centre it, but a gutter on the LEFT moves the scroller's
  // padding box in, and that is the box the browser lays the caret and
  // selection layers out from — while CodeMirror works out where to draw them
  // from the border box. Measured: the caret stood 8px right of the text it
  // belonged to, and every selection with it.
  '.cm-scroller': {
    cursor: 'text',
    fontFamily: 'var(--bw-hand)',
    lineHeight: '1.5',
    overflowY: 'auto',
    paddingInline: 'max(2rem, calc((100% - var(--bw-measure)) / 2))',
    scrollbarGutter: 'stable'
  },
  // `drawSelection` paints its own caret and selection, and CodeMirror's base
  // theme colours them for a light page (a black caret, invisible here). Both
  // take app tokens instead, so they follow light and dark with everything
  // else. `.cm-focused` is repeated to match the base rule's specificity —
  // ours is declared later, so equal specificity wins.
  // Wider than CodeMirror's 1.2px hairline: beside a hand at this size a hairline
  // reads as a rendering artefact rather than as the place the next letter goes.
  '.cm-cursor, .cm-dropCursor': { borderLeftColor: 'var(--bw-ink)', borderLeftWidth: '2px', marginLeft: '-1px' },
  // No caret while the window turns (`TURNING_ATTRIBUTE`, store.ts). The editor
  // is built edge-on and measures whenever it likes, and what it measures under
  // the turn's transform is a box squashed to nothing: measured frame by frame,
  // the caret stood out in the left margin for the whole second half and jumped
  // home when the window landed. It is drawn once there is a flat box to place
  // it in.
  ':root[data-backworkspace-turning] & .cm-cursorLayer': { visibility: 'hidden' },
  '&.cm-focused.cm-focused > .cm-scroller > .cm-selectionLayer .cm-selectionBackground, .cm-selectionBackground': {
    backgroundColor: 'var(--ui-selection-background)'
  },
  '.cm-content ::selection': { backgroundColor: 'var(--ui-selection-background)' }
})

/**
 * Hands a click in the page's margin to the line beside it.
 *
 * The centered column is the scroller's padding, so a click out there has the
 * scroller as its target — and CodeMirror listens for the mouse on the content
 * alone, which is why the page would otherwise sit there doing nothing. Its own
 * hit-testing wants no more than a height and a distance, so it answers for a
 * point in the margin exactly as it would for one on the line.
 */
export function caretFromMargin(view: EditorView, event: MouseEvent) {
  if (event.button !== 0 || event.target !== view.scrollDOM) {
    return
  }

  event.preventDefault()
  view.dispatch({ selection: { anchor: view.posAtCoords({ x: event.clientX, y: event.clientY }, false) } })
  view.focus()
}

interface BackworkspaceEditorProps {
  ariaLabel: string
  autoFocus?: boolean
  className?: string
  /** Read once at mount, like the document (the `@` list lives here). */
  extensions?: Extension
  /** The caret or selection the page was last left with. Read once at mount. */
  initialSelection?: { anchor: number; head: number }
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
  initialSelection,
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
        selection: initialSelection,
        extensions: [
          history(),
          drawSelection(),
          // Escape is left to the page: with an empty selection CodeMirror's
          // own binding reports it unhandled, so it reaches the section that
          // turns the window back.
          keymap.of([...defaultKeymap, ...historyKeymap]),
          EditorView.lineWrapping,
          EditorView.contentAttributes.of({ 'aria-label': ariaLabel, spellcheck: 'true' }),
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

    // On the scroller, not through `EditorView.domEventHandlers`: those are
    // bound to the content, which is the one element a margin click misses.
    view.scrollDOM.addEventListener('mousedown', event => caretFromMargin(view, event))

    if (autoFocus) {
      view.focus()
    }

    return () => view.destroy()
    // The document is mount-owned; re-running this would discard the user's edits.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return <div className={cn('min-h-0 flex-1 overflow-hidden', className)} ref={hostRef} />
}

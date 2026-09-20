import type { Extension } from '@codemirror/state'
import { type EditorView, ViewPlugin } from '@codemirror/view'

import { $backworkspacePage, editBackworkspacePage } from './page'

export interface PageInsertion {
  /** Where the caret goes afterwards; omitted, it stays where the reader left it. */
  caret?: number
  from: number
  insert: string
}

/** Where the reader was on a page: the caret (or selection), and how far down it was scrolled. */
export interface PagePlace {
  anchor: number
  head: number
  scrollTop: number
}

/**
 * On the document while the window is turning.
 *
 * Everything an editor measures under the turn's transform is bent: measured
 * frame by frame on a long page, the caret sat out in the margin, the page's
 * height went from 3,272px to 2,108,963px and the scroll offset to 274,488.
 * So for as long as this is set the caret is not drawn (editor.tsx), scroll
 * events are not believed, and putting the reader back where they were waits
 * for a flat box.
 */
export const TURNING_ATTRIBUTE = 'data-backworkspace-turning'

const turning = () => document.documentElement.hasAttribute(TURNING_ATTRIBUTE)

// The editor showing each owner's page, for as long as there is one.
const live = new Map<string, EditorView>()

// Where the reader left each page. An editor is built afresh every time the
// window turns to the back, and a fresh one opens at the top with its caret
// before the first letter — so someone who turned away mid-sentence came back
// to the start of the page. Kept as they go rather than read on the way out:
// by the time an editor is destroyed its DOM has left the document, and a
// detached scroller says it was never scrolled.
const places = new Map<string, PagePlace>()

/**
 * Marks the editor it is installed in as the one showing `key`'s page, and
 * keeps track of where the reader is on it.
 */
export function liveEditor(key: string): Extension {
  return ViewPlugin.define(view => {
    const keepScroll = () => {
      if (turning()) {
        return
      }

      const { anchor, head } = view.state.selection.main

      places.set(key, { anchor, head, scrollTop: view.scrollDOM.scrollTop })
    }

    live.set(key, view)
    view.scrollDOM.addEventListener('scroll', keepScroll)

    // Built flat — another profile's page swapped in while the window was
    // already over — so there is no landing to wait for.
    if (!turning()) {
      settle(key, view)
    }

    return {
      destroy() {
        view.scrollDOM.removeEventListener('scroll', keepScroll)
        live.delete(key)
      },
      update(update) {
        if (update.selectionSet || update.docChanged) {
          const { anchor, head } = update.state.selection.main

          // The scroll offset is the listener's to read: asking the DOM for it
          // here would force a layout inside every keystroke's update.
          places.set(key, { anchor, head, scrollTop: places.get(key)?.scrollTop ?? 0 })
        }
      }
    }
  })
}

/** Where `key`'s page was left, fitted to a document of `docLength` — or null the first time. */
export function placeLeftOn(key: string, docLength: number): null | PagePlace {
  const place = places.get(key)

  return place ? { ...place, anchor: Math.min(place.anchor, docLength), head: Math.min(place.head, docLength) } : null
}

// Measure in a flat box, then scroll back to where the page was left. In the
// measurement's write phase, so the offset lands on heights that are real; set
// outright rather than asked of CodeMirror, which would scroll the caret into
// view and lose how the reader had the page framed.
function settle(key: string, view: EditorView) {
  view.requestMeasure({
    read: () => null,
    write: () => {
      view.scrollDOM.scrollTop = places.get(key)?.scrollTop ?? 0
    }
  })
}

/**
 * The window has landed: have the editors on show measure themselves again and
 * go back to where they were left.
 *
 * An editor mounts while the window is edge-on, halfway through its turn. A
 * transform is not a resize, so nothing tells it when the turn is over, and
 * what it measured under it stands — the caret out in the margin, blinking —
 * until a click makes it look again. Whoever ends the turn asks.
 */
export function settleLiveEditors() {
  for (const [key, view] of live) {
    settle(key, view)
  }
}

/**
 * Write into `key`'s page, wherever that page is by now.
 *
 * What gets written arrives late — an agent's reply, a picture that had to be
 * stored first — and the window may have been turned to the front since, or
 * turned away and back. An editor owns its document from the moment it mounts:
 * text put into the store behind a mounted editor is never shown, and the
 * editor's next change saves the page without it. So the text goes through the
 * editor showing the page when there is one — whichever one that is now, not
 * the one that was there when the question left — and into the text the next
 * editor opens with when there is none.
 *
 * `place` is handed the caret only when there is an editor to have one.
 */
export function writeToPage(key: string, place: (doc: string, caret: null | number) => PageInsertion) {
  const view = live.get(key)

  if (view) {
    const { caret, from, insert } = place(view.state.doc.toString(), view.state.selection.main.head)

    view.dispatch({ changes: { from, insert }, selection: caret === undefined ? undefined : { anchor: caret } })

    return
  }

  const page = $backworkspacePage.get()

  // Only the page it was meant for: the window may be showing another
  // profile's page by now, which has nothing to do with this one's answer.
  if (page?.status === 'ready' && page.key === key) {
    const { from, insert } = place(page.content, null)
    const left = places.get(key)

    // The caret stays on the words it was on, as it would have through an
    // editor: text that lands above it pushes it down by as much. At the very
    // spot it lands, the caret stays in front — the end of a question, with
    // the reply arriving under it.
    if (left) {
      const moved = (pos: number) => (pos > from ? pos + insert.length : pos)

      places.set(key, { ...left, anchor: moved(left.anchor), head: moved(left.head) })
    }

    editBackworkspacePage(page.content.slice(0, from) + insert + page.content.slice(from))
  }
}

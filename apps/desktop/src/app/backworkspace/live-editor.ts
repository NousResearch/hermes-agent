import type { Extension } from '@codemirror/state'
import { type EditorView, ViewPlugin } from '@codemirror/view'

import { $backworkspacePage, editBackworkspacePage } from './page'

export interface PageInsertion {
  /** Where the caret goes afterwards; omitted, it stays where the reader left it. */
  caret?: number
  from: number
  insert: string
}

// The editor showing each owner's page, for as long as there is one.
const live = new Map<string, EditorView>()

/** Marks the editor it is installed in as the one showing `key`'s page. */
export function liveEditor(key: string): Extension {
  return ViewPlugin.define(view => {
    live.set(key, view)

    return { destroy: () => void live.delete(key) }
  })
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

    editBackworkspacePage(page.content.slice(0, from) + insert + page.content.slice(from))
  }
}

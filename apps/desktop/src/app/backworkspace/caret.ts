import type { EditorView } from '@codemirror/view'

/**
 * Where the caret sits, for placing something beside it.
 *
 * A position that has no rectangle — not laid out yet, or a test environment
 * without layout — falls back to the editor's own box, so whatever is being
 * placed still opens on the page rather than at its corner.
 */
export function caretRect(view: EditorView, pos: number): { bottom: number; left: number; top: number } {
  try {
    return view.coordsAtPos(pos) ?? view.dom.getBoundingClientRect()
  } catch {
    return view.dom.getBoundingClientRect()
  }
}

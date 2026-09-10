/** Collapse the selection to the end of the editor's contents.
 *
 *  The composer's undo stack, ref rendering, and trigger detection all read the
 *  live selection, and jsdom starts with none — without this they see a caret
 *  at offset 0 and take the wrong branch. */
export function placeCaretAtEnd(editor: HTMLElement) {
  const range = document.createRange()
  const selection = window.getSelection()!

  range.selectNodeContents(editor)
  range.collapse(false)
  selection.removeAllRanges()
  selection.addRange(range)
}

import { attachmentNamesForEditor, attachmentReferenceMatches } from './attachment-references'
import {
  appendComposerContents,
  composerPlainText,
  composerTextBefore,
  insertComposerContentsAtCaret,
  placeCaretAtOffset,
  renderComposerContents
} from './rich-editor'

/** Restyle only when reference spans changed; ordinary typing never repaints. */
export function refreshAttachmentReferences(editor: HTMLElement) {
  const text = composerPlainText(editor)
  const expected = attachmentReferenceMatches(text, attachmentNamesForEditor(editor))
  const actual = [...editor.querySelectorAll<HTMLElement>('[data-attachment-reference]')]

  if (
    actual.length === expected.length &&
    actual.every((chip, index) => {
      const offset = Array.prototype.indexOf.call(chip.parentNode!.childNodes, chip)

      return (
        chip.dataset.refText === expected[index].text &&
        composerTextBefore(editor, chip.parentNode!, offset).length === expected[index].start
      )
    })
  ) {
    return
  }

  const selection = window.getSelection()

  const ownsSelection =
    selection?.anchorNode &&
    selection.focusNode &&
    editor.contains(selection.anchorNode) &&
    editor.contains(selection.focusNode)

  const anchor = ownsSelection ? composerTextBefore(editor, selection.anchorNode!, selection.anchorOffset).length : null

  const focus = ownsSelection ? composerTextBefore(editor, selection.focusNode!, selection.focusOffset).length : null
  renderComposerContents(editor, text)

  if (anchor !== null && focus !== null && selection) {
    placeCaretAtOffset(editor, anchor)
    const node = selection.anchorNode!
    const offset = selection.anchorOffset
    placeCaretAtOffset(editor, focus)
    selection.setBaseAndExtent(node, offset, selection.focusNode!, selection.focusOffset)
  }
}

export function insertAttachmentReference(editor: HTMLElement, name: string) {
  const selection = window.getSelection()
  const range = selection?.rangeCount ? selection.getRangeAt(0) : null
  const ownsSelection = range && editor.contains(range.commonAncestorContainer)
  const text = composerPlainText(editor)
  const before = ownsSelection ? composerTextBefore(editor, range.startContainer, range.startOffset) : text
  const end = ownsSelection ? composerTextBefore(editor, range.endContainer, range.endOffset).length : text.length
  const prefix = before && !/\s$/.test(before) ? ' ' : ''
  const suffix = !text[end] || !/\s/.test(text[end]) ? ' ' : ''
  const reference = `${prefix}[${name}]${suffix}`

  if (ownsSelection) {
    insertComposerContentsAtCaret(editor, reference)
  } else {
    // A background attach must not move another editor's document selection.
    appendComposerContents(editor, reference, { attachmentNames: attachmentNamesForEditor(editor) })
  }
}

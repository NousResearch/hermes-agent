import type { KeyboardEvent } from 'react'

import { type ExtraListStyle, type ListEdit, listEdit, type ListKey } from '@/lib/markdown-lists'

import {
  caretOffsetInEditor,
  composerPlainText,
  placeCaretAtOffset,
  renderComposerContents,
  revealCaret
} from './rich-editor'

/** The list meaning of a keydown, if it has one. Plain Enter is not here: it always sends. */
export function composerListKey(event: KeyboardEvent<HTMLElement>): ListKey | null {
  if (event.metaKey || event.ctrlKey || event.altKey) {
    return null
  }

  switch (event.key) {
    case 'Enter':
      return event.shiftKey ? 'newline' : null

    case 'Tab':
      return event.shiftKey ? 'outdent' : 'indent'

    case 'Backspace':
      return event.shiftKey ? null : 'backspace'

    default:
      return null
  }
}

/** The list edit `key` makes at the caret, or null when the key keeps its normal meaning. */
export function listEditAtCaret(
  editor: HTMLElement,
  key: ListKey,
  styles: ReadonlySet<ExtraListStyle>
): ListEdit | null {
  const selection = window.getSelection()
  const range = selection?.rangeCount ? selection.getRangeAt(0) : null

  if (!range?.collapsed || !editor.contains(range.startContainer)) {
    return null
  }

  return listEdit(composerPlainText(editor), caretOffsetInEditor(editor), key, styles)
}

/** Paint `edit` into the editor and keep the caret in view. False when it changes nothing. */
export function applyListEdit(editor: HTMLElement, edit: ListEdit): boolean {
  if (edit.text === composerPlainText(editor)) {
    return false
  }

  renderComposerContents(editor, edit.text)

  // A caret on an empty last line needs a placeholder <br> to stand on, or
  // Chromium moves it to the end of the line above. Native Shift+Enter adds the
  // same placeholder, and Chromium drops it as soon as the line gets text.
  if (edit.caret === edit.text.length && edit.text.endsWith('\n')) {
    editor.append(document.createElement('br'))
  }

  placeCaretAtOffset(editor, edit.caret)
  // A rebuilt editor keeps its old scrollTop, so a new last line of a long
  // draft would land below the fold (the same gap paste and voice close).
  revealCaret(editor)

  return true
}

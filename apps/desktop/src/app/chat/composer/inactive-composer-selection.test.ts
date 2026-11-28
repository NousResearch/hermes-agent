import { expect, it, vi } from 'vitest'

import { composerPlainText, normalizeComposerEditorDom, RICH_INPUT_SLOT } from './rich-editor'

it('normalizes an unfocused composer without querying the document selection', () => {
  const editor = document.createElement('div')
  editor.dataset.slot = RICH_INPUT_SLOT
  editor.contentEditable = 'true'
  editor.append(document.createTextNode('draft'), document.createTextNode(''))
  document.body.append(editor)
  const selection = vi.spyOn(window, 'getSelection')
  try {
    normalizeComposerEditorDom(editor)
    expect(selection).not.toHaveBeenCalled()
    expect(composerPlainText(editor)).toBe('draft')
    expect(editor.childNodes).toHaveLength(1)
  } finally {
    selection.mockRestore()
    editor.remove()
  }
})

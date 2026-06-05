import { describe, expect, it, vi } from 'vitest'

import { composerPlainText, renderComposerContents, RICH_INPUT_SLOT, syncComposerDraft } from './rich-editor'

describe('syncComposerDraft', () => {
  function editorWithText(text: string) {
    const editor = document.createElement('div')
    editor.dataset.slot = RICH_INPUT_SLOT
    editor.textContent = text

    return editor
  }

  it('commits the live editor text when the tracked draft is stale (IME compositionend, dropped input event)', () => {
    // Reproduces #39025: the contentEditable holds IME-committed text but the
    // tracked draft is still empty because the post-composition input event
    // never updated it.
    const editor = editorWithText('你好世界')
    const setText = vi.fn()

    const next = syncComposerDraft(editor, '', setText)

    expect(next).toBe('你好世界')
    expect(setText).toHaveBeenCalledTimes(1)
    expect(setText).toHaveBeenCalledWith('你好世界')
  })

  it('is a no-op when the tracked draft already matches the editor', () => {
    const editor = editorWithText('hello')
    const setText = vi.fn()

    expect(syncComposerDraft(editor, 'hello', setText)).toBe('hello')
    expect(setText).not.toHaveBeenCalled()
  })

  it('returns the previous draft and does not write when there is no editor', () => {
    const setText = vi.fn()

    expect(syncComposerDraft(null, 'keep', setText)).toBe('keep')
    expect(setText).not.toHaveBeenCalled()
  })
})

describe('renderComposerContents', () => {
  it('renders refs and raw text without interpreting user text as HTML', () => {
    const editor = document.createElement('div')
    editor.dataset.slot = RICH_INPUT_SLOT

    renderComposerContents(editor, '@file:`<img src=x onerror=alert(1)>` <b>raw</b>')

    expect(editor.querySelector('img')).toBeNull()
    expect(editor.querySelector('b')).toBeNull()
    expect(editor.textContent).toContain('<img src=x onerror=alert(1)>')
    expect(editor.textContent).toContain('<b>raw</b>')
    expect(composerPlainText(editor)).toBe('@file:`<img src=x onerror=alert(1)>` <b>raw</b>')
  })
})

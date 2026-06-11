import { describe, expect, it } from 'vitest'

import { composerPlainText, renderComposerContents, RICH_INPUT_SLOT } from './rich-editor'

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

describe('composerPlainText', () => {
  it('does not add a trailing newline for browser-created block wrappers', () => {
    const editor = document.createElement('div')
    const line = document.createElement('div')

    editor.dataset.slot = RICH_INPUT_SLOT
    line.textContent = 'help me debug this'
    editor.append(line)

    expect(composerPlainText(editor)).toBe('help me debug this')
  })

  it('keeps newlines between browser-created block wrappers', () => {
    const editor = document.createElement('div')
    const first = document.createElement('div')
    const second = document.createElement('div')

    editor.dataset.slot = RICH_INPUT_SLOT
    first.textContent = 'first line'
    second.textContent = 'second line'
    editor.append(first, second)

    expect(composerPlainText(editor)).toBe('first line\nsecond line')
  })
})

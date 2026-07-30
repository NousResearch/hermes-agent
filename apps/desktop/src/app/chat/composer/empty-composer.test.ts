import { describe, expect, it } from 'vitest'

import { composerPlainText, normalizeComposerEditorDom, RICH_INPUT_SLOT } from './rich-editor'

function editor(): HTMLDivElement {
  const el = document.createElement('div')

  el.dataset.slot = RICH_INPUT_SLOT
  el.contentEditable = 'true'
  document.body.append(el)

  return el
}

/** Whatever emptied it — Delete, cut, Chromium's own selection-delete — the
 *  normalizer lands on the same DOM. */
function emptied(): HTMLDivElement {
  const el = editor()

  el.append(document.createTextNode('hello'))
  el.replaceChildren()
  normalizeComposerEditorDom(el)

  return el
}

describe('an emptied composer reads as empty', () => {
  it('keeps the placeholder <br> so the contenteditable holds its height', () => {
    // The scaffolding is deliberate: a childless contenteditable collapses to a
    // sliver in Chromium. It just must not read as content.
    expect(emptied().innerHTML).toBe('<br>')
  })

  it('reads that editor as empty, not as a newline', () => {
    expect(composerPlainText(emptied())).toBe('')
  })

  it('reads a truly childless editor as empty', () => {
    expect(composerPlainText(editor())).toBe('')
  })

  it('still reads a real Shift+Enter line break as a newline', () => {
    const el = editor()

    el.append(document.createTextNode('one'), document.createElement('br'), document.createTextNode('two'))

    expect(composerPlainText(el)).toBe('one\ntwo')
  })

  it('still reads a trailing break after text as a newline', () => {
    const el = editor()

    el.append(document.createTextNode('one'), document.createElement('br'))

    expect(composerPlainText(el)).toBe('one\n')
  })

  it('only treats the EDITOR\u2019s lone <br> as scaffolding, not a nested one', () => {
    // A lone <br> inside some other element is a real line break; the exemption
    // is scoped to the editor root by its slot marker. (The block wrapper adds
    // its own trailing newline — unchanged behavior, asserted so the exemption
    // can't quietly widen to nested nodes.)
    const el = editor()
    const inner = document.createElement('div')

    inner.append(document.createElement('br'))
    el.append(document.createTextNode('one'), inner)

    expect(composerPlainText(el)).toBe('one\n\n')
  })
})

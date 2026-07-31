import { describe, expect, it } from 'vitest'

import {
  composerPlainText,
  normalizeComposerEditorDom,
  renderComposerContents,
  RICH_INPUT_SLOT
} from './rich-editor'

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

/** The rule the stylesheet paints the placeholder with. `:empty` alone goes
 *  false the instant the scaffolding <br> lands. */
const PLACEHOLDER_SHOWS = ':is(:empty, [data-empty])'

describe('an emptied composer shows its placeholder again', () => {
  it('advertises emptiness once the scaffolding break is in place', () => {
    expect(emptied().matches(PLACEHOLDER_SHOWS)).toBe(true)
  })

  it('advertises emptiness for a truly childless editor', () => {
    expect(editor().matches(PLACEHOLDER_SHOWS)).toBe(true)
  })

  it('stops advertising it once something is typed', () => {
    const el = emptied()

    el.replaceChildren(document.createTextNode('hi'))
    normalizeComposerEditorDom(el)

    expect(el.matches(PLACEHOLDER_SHOWS)).toBe(false)
  })

  // A text node is invisible to selectors, so `one<br>` and `<br>` are the same
  // shape to any pure-CSS rule (`:has(> br:only-child)` matches both and paints
  // the placeholder straight over the user's text). The DOM writer has to say.
  it('does not advertise emptiness for a trailing break after text', () => {
    const el = editor()

    el.append(document.createTextNode('one'), document.createElement('br'))
    normalizeComposerEditorDom(el)

    expect(el.matches(PLACEHOLDER_SHOWS)).toBe(false)
  })

  it('does not advertise emptiness for a Shift+Enter break between text', () => {
    const el = editor()

    el.append(document.createTextNode('one'), document.createElement('br'), document.createTextNode('two'))
    normalizeComposerEditorDom(el)

    expect(el.matches(PLACEHOLDER_SHOWS)).toBe(false)
  })

  // Repainting from text (restored draft, undo, completion rebuild) is the
  // other writer that reshapes the editor root — it must not strand the marker.
  it('drops the marker when a draft is painted back in', () => {
    const el = emptied()

    renderComposerContents(el, 'restored draft')

    expect(el.matches(PLACEHOLDER_SHOWS)).toBe(false)
  })

  it('re-advertises emptiness when a draft is painted back out', () => {
    const el = editor()

    renderComposerContents(el, 'temporary')
    renderComposerContents(el, '')

    expect(el.matches(PLACEHOLDER_SHOWS)).toBe(true)
  })
})

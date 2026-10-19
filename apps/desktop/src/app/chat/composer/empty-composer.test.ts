import { describe, expect, it } from 'vitest'

import {
  beginComposerComposition,
  composerPlainText,
  deleteChipBeforeCaret,
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

  // Chromium leaves zero-length text nodes behind whenever an edit lands next
  // to a contenteditable=false chip. They render as nothing, so an editor
  // holding only those is empty to the user — counting them as contents left
  // the placeholder hidden under a composer that looked blank.
  it('advertises emptiness for an editor holding only zero-length text nodes', () => {
    const el = editor()

    el.append(document.createTextNode(''), document.createTextNode(''))
    normalizeComposerEditorDom(el)

    expect(el.matches(PLACEHOLDER_SHOWS)).toBe(true)
  })

  it('does not advertise emptiness while real text sits beside that litter', () => {
    const el = editor()

    el.append(document.createTextNode(''), document.createTextNode('one'), document.createTextNode(''))
    normalizeComposerEditorDom(el)

    expect(el.matches(PLACEHOLDER_SHOWS)).toBe(false)
  })

  // Input events are skipped for the duration of an IME composition, so nothing
  // else clears the marker until it ends — the hint would sit behind the
  // hiragana the user is composing (#75960).
  it('hides the placeholder before IME preedit text starts', () => {
    const el = emptied()

    beginComposerComposition(el)

    expect(el.matches(PLACEHOLDER_SHOWS)).toBe(false)
  })

  it('brings the placeholder back when composition ends with nothing committed', () => {
    const el = emptied()

    beginComposerComposition(el)
    normalizeComposerEditorDom(el)

    expect(el.matches(PLACEHOLDER_SHOWS)).toBe(true)
  })
})

/** A directive chip, as `refChipElement` builds it. */
function chip(): HTMLSpanElement {
  const el = document.createElement('span')

  el.contentEditable = 'false'
  el.dataset.refText = '@folder:`apps/desktop/`'
  el.append(document.createTextNode('apps/desktop/'))

  return el
}

function caretAt(node: Node, offset: number) {
  const range = document.createRange()

  range.setStart(node, offset)
  range.collapse(true)

  const selection = window.getSelection()

  selection?.removeAllRanges()
  selection?.addRange(range)
}

/** Committing a completion empties the typed token's text node instead of
 *  removing it, and `Range.insertNode` splits the line around the caret — so a
 *  freshly-chipped directive sits between zero-length text nodes. Backspace has
 *  to see past them or the chip can't be deleted at all. */
describe('backspace deletes a chip surrounded by Chromium litter', () => {
  it('deletes the chip when the caret sits in a zero-length text node after it', () => {
    const el = editor()

    el.append(document.createTextNode(''), chip(), document.createTextNode(''))
    caretAt(el.childNodes[2] as Node, 0)

    expect(deleteChipBeforeCaret(el)).toBe(true)
    expect(el.querySelector('[data-ref-text]')).toBeNull()
  })

  it('deletes the chip when the caret is past a zero-length text node at editor level', () => {
    const el = editor()

    el.append(chip(), document.createTextNode(''))
    caretAt(el, 2)

    expect(deleteChipBeforeCaret(el)).toBe(true)
    expect(el.querySelector('[data-ref-text]')).toBeNull()
  })

  it('still swallows the auto-inserted trailing space through that litter', () => {
    const el = editor()

    el.append(chip(), document.createTextNode(''), document.createTextNode(' '))
    caretAt(el, 2)

    expect(deleteChipBeforeCaret(el)).toBe(true)
    expect(composerPlainText(el)).toBe('')
  })

  it('keeps real following text when it deletes the chip', () => {
    const el = editor()

    el.append(chip(), document.createTextNode(''), document.createTextNode(' and this'))
    caretAt(el, 2)

    expect(deleteChipBeforeCaret(el)).toBe(true)
    expect(composerPlainText(el)).toBe('and this')
  })

  it('leaves plain text to the native backspace', () => {
    const el = editor()

    el.append(document.createTextNode('hello'))
    caretAt(el.firstChild as Node, 5)

    expect(deleteChipBeforeCaret(el)).toBe(false)
  })

  it('sweeps the litter out of the editor when it normalizes', () => {
    const el = editor()

    el.append(document.createTextNode(''), chip(), document.createTextNode(''))
    normalizeComposerEditorDom(el)

    expect(Array.from(el.childNodes).map(node => node.nodeName)).toEqual(['SPAN'])
  })
})

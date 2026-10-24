import { afterEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'

import { COMPOSER_AREAS, type ComposerNonProseProvider, type ComposerNonProseRange } from './contrib'
import { COMPOSER_NON_PROSE_PER_PROVIDER, COMPOSER_NON_PROSE_TOTAL } from './non-prose'
import {
  beginComposerComposition,
  caretOffsetInEditor,
  composerPlainText,
  normalizeComposerEditorDom,
  renderComposerContents,
  RICH_INPUT_SLOT
} from './rich-editor'

const disposers: Array<() => void> = []

function addProvider(id: string, ranges: ComposerNonProseProvider['ranges']) {
  disposers.push(
    registry.register({ area: COMPOSER_AREAS.nonProse, data: { ranges } satisfies ComposerNonProseProvider, id })
  )
}

/** A composer surface that has opted into spell checking — the seam's gate. */
function editorFor(text: string, spellcheck = true): HTMLDivElement {
  const editor = document.createElement('div')

  editor.dataset.slot = RICH_INPUT_SLOT
  editor.contentEditable = 'true'
  editor.setAttribute('spellcheck', spellcheck ? 'true' : 'false')
  renderComposerContents(editor, text)

  return editor
}

const wrappers = (editor: HTMLElement) => Array.from(editor.querySelectorAll<HTMLElement>('[data-composer-nonprose]'))

const texts = (editor: HTMLElement) => wrappers(editor).map(wrapper => wrapper.textContent)

afterEach(() => {
  disposers.splice(0).forEach(dispose => dispose())
  document.body.replaceChildren()
  vi.restoreAllMocks()
})

describe('composer.nonProse — rendering', () => {
  it('is reachable at the documented area id', () => {
    expect(COMPOSER_AREAS.nonProse).toBe('composer.nonProse')
  })

  it('paints a declared range as a spellcheck="false" wrapper with its exact text', () => {
    addProvider('spellcheck', ({ text }) => [{ end: text.indexOf('`', 1) + 1, kind: 'code', start: 0 }])

    const editor = editorFor('`npm run build` then ship')

    expect(texts(editor)).toEqual(['`npm run build`'])

    const [wrapper] = wrappers(editor)

    expect(wrapper.tagName).toBe('SPAN')
    expect(wrapper.getAttribute('spellcheck')).toBe('false')
    expect(wrapper.getAttribute('data-composer-nonprose')).toBe('code')
    // Editable and unstyled: the attribute is the whole hook.
    expect(wrapper.hasAttribute('contenteditable')).toBe(false)
    expect(wrapper.getAttribute('style')).toBeNull()
    expect(wrapper.getAttribute('class')).toBeNull()
  })

  it('paints an unkinded range as `code` and ignores an unusable kind', () => {
    addProvider('spellcheck', ({ text }) => [
      { end: 1, start: 0 },
      { end: 3, kind: 'shouty' as never, start: 2 }
    ])

    expect(wrappers(editorFor('abcd')).map(wrapper => wrapper.dataset.composerNonprose)).toEqual(['code', 'code'])
  })

  it('covers the whole draft, and every line of a multi-line one, without straddling a <br>', () => {
    addProvider('spellcheck', ({ text }) => [{ end: text.length, start: 0 }])

    const single = editorFor('const answer = 42')

    expect(single.childNodes).toHaveLength(1)
    expect(texts(single)).toEqual(['const answer = 42'])
    expect(composerPlainText(single)).toBe('const answer = 42')

    const multi = editorFor('const answer = 42\nconst other = 7')

    expect(texts(multi)).toEqual(['const answer = 42', 'const other = 7'])
    expect(multi.querySelectorAll('[data-composer-nonprose] br')).toHaveLength(0)
    expect(composerPlainText(multi)).toBe('const answer = 42\nconst other = 7')
  })

  it('never swallows a chip: the range is clipped around the ref it covers', () => {
    addProvider('spellcheck', ({ text }) => [{ end: text.length, start: 0 }])

    const editor = editorFor('pre @file:`src/a.ts` tail')

    expect(editor.querySelector('[data-ref-text]')).not.toBeNull()
    expect(editor.querySelectorAll('[data-composer-nonprose] [data-ref-text]')).toHaveLength(0)
    expect(texts(editor)).toEqual(['pre ', ' tail'])
    expect(composerPlainText(editor)).toBe('pre @file:`src/a.ts` tail')
  })

  it('drops whitespace-only, empty, inverted, fractional and out-of-range declarations', () => {
    addProvider('spellcheck', () => [
      { end: 2, start: 2 }, // empty
      { end: 1, start: 4 }, // inverted
      { end: 3.5, start: 1 }, // not an index
      { end: 1, start: 0 } // the only valid one
    ])

    const editor = editorFor('a  b')

    expect(texts(editor)).toEqual(['a'])
  })
})

describe('composer.nonProse — zero cost when unused', () => {
  it('leaves the DOM byte-identical with no provider registered', () => {
    const editor = editorFor('const x = 1\nnext line')

    expect(editor.innerHTML).toBe('const x = 1<br>next line')
    expect(editor.querySelectorAll('[data-composer-nonprose]')).toHaveLength(0)
    expect(editor.childNodes).toHaveLength(3)
  })

  it('skips the providers entirely on a surface that has not opted into spell check', () => {
    const ranges = vi.fn(() => [{ end: 3, start: 0 }])

    addProvider('spellcheck', ranges)

    const editor = editorFor('abcdef', false)

    expect(ranges).not.toHaveBeenCalled()
    expect(editor.innerHTML).toBe('abcdef')
    expect(editor.querySelectorAll('[data-composer-nonprose]')).toHaveLength(0)
  })

  it('reads the inherited attribute, the state the shipped composer is actually in', () => {
    const ranges = vi.fn(() => [{ end: 3, start: 0 }])

    addProvider('spellcheck', ranges)

    // Both composers hard-code `spellCheck={false}` (#44415) on the editor, and
    // `spellcheck` inherits — so this is what every user gets today, until
    // `host.composer.setSpellcheck` (#143057) can set the attribute.
    const host = document.createElement('div')

    host.setAttribute('spellcheck', 'false')

    const editor = document.createElement('div')

    editor.dataset.slot = RICH_INPUT_SLOT
    editor.contentEditable = 'true'
    host.append(editor)
    document.body.append(host)
    renderComposerContents(editor, 'abcdef')

    expect(ranges).not.toHaveBeenCalled()
    expect(editor.innerHTML).toBe('abcdef')
    expect(editor.querySelectorAll('[data-composer-nonprose]')).toHaveLength(0)
  })

  it('does not consult a provider whose contribution carries no ranges function', () => {
    disposers.push(registry.register({ area: COMPOSER_AREAS.nonProse, data: { nope: true }, id: 'junk' }))

    expect(() => editorFor('abcdef')).not.toThrow()
    expect(wrappers(editorFor('abcdef'))).toHaveLength(0)
  })
})

describe('composer.nonProse — resolution', () => {
  it('hands the provider the draft text and the surface’s session address', () => {
    const seen: Array<{ sessionId: string; text: string }> = []

    addProvider('spellcheck', context => {
      seen.push({ sessionId: context.sessionId, text: context.text })

      return []
    })

    const editor = document.createElement('div')

    editor.dataset.slot = RICH_INPUT_SLOT
    editor.spellcheck = true
    editor.dataset.composerSession = 'session-7'
    renderComposerContents(editor, 'hello')

    expect(seen).toEqual([{ sessionId: 'session-7', text: 'hello' }])

    // A surface with no address of its own still reaches the provider.
    renderComposerContents(editorFor('hello'), 'hello')

    expect(seen.at(-1)?.sessionId).toBe('')
  })

  it('merges overlapping and adjacent ranges from two providers into one wrapper', () => {
    addProvider('first', () => [{ end: 6, start: 0 }])
    addProvider('second', () => [
      { end: 10, start: 4 },
      { end: 13, start: 10 }
    ])

    const editor = editorFor('abcdefghijklm')

    expect(texts(editor)).toEqual(['abcdefghijklm'])
  })

  it('isolates a throwing provider and keeps every other provider’s ranges', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined)

    addProvider('boom', () => {
      throw new Error('broken plugin')
    })
    addProvider('ok', () => [{ end: 3, start: 0 }])

    const editor = editorFor('abcdef')

    expect(texts(editor)).toEqual(['abc'])
    expect(warn).toHaveBeenCalled()
  })

  it('caps each provider at 64 ranges and one pass at 256', () => {
    const text = 'abc'.repeat(400)

    const spaced = (from: number, count: number): ComposerNonProseRange[] =>
      Array.from({ length: count }, (_, index) => ({ end: (from + index) * 3 + 1, start: (from + index) * 3 }))

    addProvider('greedy', () => spaced(0, 200))

    expect(wrappers(editorFor(text))).toHaveLength(COMPOSER_NON_PROSE_PER_PROVIDER)

    disposers.splice(0).forEach(dispose => dispose())

    for (const [index, id] of ['one', 'two', 'three', 'four', 'five'].entries()) {
      addProvider(id, () => spaced(index * COMPOSER_NON_PROSE_PER_PROVIDER, 200))
    }

    expect(wrappers(editorFor(text))).toHaveLength(COMPOSER_NON_PROSE_TOTAL)
  })
})

describe('composer.nonProse — the reconciler owns the wrapper', () => {
  it('round-trips to the original text and is never read as a chip', () => {
    addProvider('spellcheck', ({ text }) => [{ end: text.length, start: 0 }])

    const editor = editorFor('see `npm i` now')

    const [wrapper] = wrappers(editor)

    expect(wrapper.hasAttribute('data-ref-text')).toBe(false)
    expect(wrapper.dataset.refText).toBeUndefined()
    expect(composerPlainText(editor)).toBe('see `npm i` now')

    // A wrapper ending the draft keeps a real line break after it: the
    // trailing-<br> rule only drops one after a chip.
    const trailing = editorFor('`code`')

    trailing.append(document.createElement('br'))
    normalizeComposerEditorDom(trailing)

    expect(texts(trailing)).toEqual(['`code`'])
    expect(trailing.querySelector('br')).not.toBeNull()
    expect(composerPlainText(trailing)).toBe('`code`\n')
  })

  it('unwraps a range the provider no longer covers, keeping the caret in place', () => {
    let ranges: ComposerNonProseRange[] = [{ end: 5, start: 0 }]

    addProvider('spellcheck', () => ranges)

    const editor = editorFor('alpha beta')

    document.body.append(editor)

    const [wrapper] = wrappers(editor)

    expect(wrapper.textContent).toBe('alpha')

    const text = wrapper.firstChild as Text
    const selection = window.getSelection()!

    selection.setBaseAndExtent(text, 3, text, 3)
    ranges = []
    normalizeComposerEditorDom(editor)

    expect(wrappers(editor)).toHaveLength(0)
    expect(composerPlainText(editor)).toBe('alpha beta')
    expect(selection.isCollapsed).toBe(true)
    expect(selection.anchorNode).toBe(text)
    expect(selection.anchorOffset).toBe(3)
  })

  it('keeps a still-covered wrapper through a reconcile', () => {
    addProvider('spellcheck', ({ text }) => [{ end: text.length, start: 0 }])

    const editor = editorFor('keep me')

    normalizeComposerEditorDom(editor)

    expect(texts(editor)).toEqual(['keep me'])
    expect(composerPlainText(editor)).toBe('keep me')
  })

  it('unwraps the wrappers when the provider unloads', () => {
    addProvider('spellcheck', ({ text }) => [{ end: text.length, start: 0 }])

    const editor = editorFor('plugin text')

    expect(wrappers(editor)).toHaveLength(1)

    disposers.splice(0).forEach(dispose => dispose())
    normalizeComposerEditorDom(editor)

    expect(wrappers(editor)).toHaveLength(0)
    expect(editor.innerHTML).toBe('plugin text')
  })

  it('unwraps a wrapper core would never emit — one holding a chip', () => {
    addProvider('spellcheck', ({ text }) => [{ end: text.length, start: 0 }])

    const editor = editorFor('@file:`src/a.ts` tail')
    const [wrapper] = wrappers(editor)

    // Simulate the shape a stale writer (or a sloppy paste) leaves behind.
    const chip = editor.querySelector('[data-ref-text]')!

    wrapper.prepend(chip)
    normalizeComposerEditorDom(editor)

    expect(editor.querySelectorAll('[data-composer-nonprose] [data-ref-text]')).toHaveLength(0)
    expect(composerPlainText(editor)).toBe('@file:`src/a.ts` tail')
  })

  it('defers re-decoration while an IME composition is in flight', () => {
    let ranges: ComposerNonProseRange[] = [{ end: 5, start: 0 }]

    addProvider('spellcheck', () => ranges)

    const editor = editorFor('hello world')

    document.body.append(editor)
    beginComposerComposition(editor)
    ranges = []
    normalizeComposerEditorDom(editor)

    expect(wrappers(editor)).toHaveLength(1)

    // Chromium fires compositionend before the composer's own flush handler.
    editor.dispatchEvent(new Event('compositionend'))
    normalizeComposerEditorDom(editor)

    expect(wrappers(editor)).toHaveLength(0)
    expect(composerPlainText(editor)).toBe('hello world')
  })

  it('cannot be wedged by a missed compositionend: blur ends the deferral too', () => {
    let ranges: ComposerNonProseRange[] = [{ end: 5, start: 0 }]

    addProvider('spellcheck', () => ranges)

    const editor = editorFor('hello world')

    document.body.append(editor)
    beginComposerComposition(editor)
    ranges = []

    // Focus jumps, an input-source switch or a programmatic DOM swap abort a
    // composition without Chromium ever firing compositionend (#44135). The
    // marker must not outlive the focus it belonged to.
    editor.dispatchEvent(new FocusEvent('blur'))
    normalizeComposerEditorDom(editor)

    expect(wrappers(editor)).toHaveLength(0)

    // And the next pass is not deferred either — the wedge used to persist for
    // the rest of the editor's life.
    ranges = [{ end: 5, start: 0 }]
    normalizeComposerEditorDom(editor)

    expect(texts(editor)).toEqual(['hello'])
  })
})

describe('composer.nonProse — the caret survives the un-wrap', () => {
  /**
   * Serialized offset of an arbitrary DOM endpoint, in the same units
   * `caretOffsetInEditor` reports.
   */
  function offsetOf(editor: HTMLElement, node: Node, offset: number): number {
    const probe = document.createRange()

    probe.selectNodeContents(editor)
    probe.setEnd(node, offset)

    const scratch = document.createElement('div')

    scratch.dataset.slot = RICH_INPUT_SLOT
    scratch.append(probe.cloneContents())

    return composerPlainText(scratch).length
  }

  /** Split the wrapper's text into one node per character — the shape Chromium
   *  leaves after a few edits inside the span, and the shape that makes an
   *  endpoint's child index meaningless across the un-wrap. */
  function fray(wrapper: HTMLElement) {
    wrapper.replaceChildren(...Array.from(wrapper.textContent ?? '').map(ch => document.createTextNode(ch)))
  }

  it('keeps a caret anchored at the END of the wrapper element', () => {
    let ranges: ComposerNonProseRange[] = [{ end: 5, start: 0 }]

    addProvider('spellcheck', () => ranges)

    const editor = editorFor('alpha beta')

    document.body.append(editor)

    const [wrapper] = wrappers(editor)

    fray(wrapper)

    const selection = window.getSelection()!
    const range = document.createRange()

    range.selectNodeContents(wrapper)
    range.collapse(false)
    selection.removeAllRanges()
    selection.addRange(range)

    expect(caretOffsetInEditor(editor)).toBe(5)

    ranges = []
    normalizeComposerEditorDom(editor)

    expect(wrappers(editor)).toHaveLength(0)
    expect(composerPlainText(editor)).toBe('alpha beta')
    // Used to land at 0: the endpoint was "on the wrapper", the old restore
    // mapped that to the editor index where the wrapper STARTED.
    expect(caretOffsetInEditor(editor)).toBe(5)
  })

  it('keeps a caret anchored on the editor just AFTER a frayed wrapper', () => {
    let ranges: ComposerNonProseRange[] = [{ end: 5, start: 0 }]

    addProvider('spellcheck', () => ranges)

    const editor = editorFor('alpha beta')

    document.body.append(editor)

    const [wrapper] = wrappers(editor)

    fray(wrapper)

    const selection = window.getSelection()!
    const range = document.createRange()

    // The shape `placeCaretEnd`, Ctrl+A and a click below the last line leave.
    range.setStart(editor, 1)
    range.collapse(true)
    selection.removeAllRanges()
    selection.addRange(range)

    expect(caretOffsetInEditor(editor)).toBe(5)

    ranges = []
    normalizeComposerEditorDom(editor)

    expect(wrappers(editor)).toHaveLength(0)
    expect(composerPlainText(editor)).toBe('alpha beta')
    expect(caretOffsetInEditor(editor)).toBe(5)
  })

  it('maps a parent-anchored endpoint while the other end stays inside the wrapper', () => {
    let ranges: ComposerNonProseRange[] = [{ end: 5, start: 0 }]

    addProvider('spellcheck', () => ranges)

    const editor = editorFor('alpha beta')

    document.body.append(editor)

    const [wrapper] = wrappers(editor)

    wrapper.replaceChildren(document.createTextNode('al'), document.createTextNode('pha'))

    const inside = wrapper.lastChild as Text
    const selection = window.getSelection()!

    // A backwards selection: the *anchor* is the parent-anchored one after the
    // wrapper, the focus is inside it. The old restore re-applied the anchor's
    // pre-unwrap child index to the post-unwrap child list.
    selection.setBaseAndExtent(editor, 1, inside, 2)

    expect(offsetOf(editor, selection.anchorNode!, selection.anchorOffset)).toBe(5)

    ranges = []
    normalizeComposerEditorDom(editor)

    expect(wrappers(editor)).toHaveLength(0)
    expect(composerPlainText(editor)).toBe('alpha beta')
    expect(offsetOf(editor, selection.anchorNode!, selection.anchorOffset)).toBe(5)
    expect(offsetOf(editor, selection.focusNode!, selection.focusOffset)).toBe(4)
  })
})

describe('composer.nonProse — the caps are caps', () => {
  const lines = Array.from({ length: 400 }, (_, index) => `line ${index}`)
  const text = lines.join('\n')

  it('counts PIECES against the per-provider budget, not declarations', () => {
    // One declaration, one range object — 400 wrappers if the split is not
    // counted. The documented ceiling is COMPOSER_NON_PROSE_PER_PROVIDER.
    addProvider('greedy', ({ text: draft }) => [{ end: draft.length, start: 0 }])

    expect(wrappers(editorFor(text)).length).toBe(COMPOSER_NON_PROSE_PER_PROVIDER)
  })

  it('does not let a greedy provider starve another out of the shared budget', () => {
    addProvider('greedy', ({ text: draft }) => [{ end: draft.length, start: 0 }])
    addProvider('tail', ({ text: draft }) => [{ end: draft.length, start: draft.length - 6 }])

    const editor = editorFor(text)
    const tail = text.slice(-6)

    // 64 from the greedy provider plus the tail's one: the greedy one cannot
    // take the tail's turn.
    expect(wrappers(editor)).toHaveLength(COMPOSER_NON_PROSE_PER_PROVIDER + 1)
    expect(wrappers(editor).length).toBeLessThanOrEqual(COMPOSER_NON_PROSE_TOTAL)
    expect(composerPlainText(editor)).toBe(text)
    expect(wrappers(editor).some(wrapper => wrapper.textContent === tail)).toBe(true)
  })

  it('still spends the whole budget when the providers have enough to fill it', () => {
    const spaced = (from: number, count: number): ComposerNonProseRange[] =>
      Array.from({ length: count }, (_, index) => ({ end: (from + index) * 3 + 1, start: (from + index) * 3 }))

    for (const [index, id] of ['one', 'two', 'three', 'four', 'five'].entries()) {
      addProvider(id, () => spaced(index * COMPOSER_NON_PROSE_PER_PROVIDER, 200))
    }

    expect(wrappers(editorFor('abc'.repeat(400)))).toHaveLength(COMPOSER_NON_PROSE_TOTAL)
  })
})

describe('composer.nonProse — the reconciler re-paints', () => {
  it('re-paints a wrapper the provider asks for again after a miss', () => {
    let ranges: ComposerNonProseRange[] = [{ end: 5, start: 0 }]

    addProvider('spellcheck', () => ranges)

    const editor = editorFor('alpha beta')

    expect(texts(editor)).toEqual(['alpha'])

    ranges = []
    normalizeComposerEditorDom(editor)

    expect(wrappers(editor)).toHaveLength(0)

    ranges = [{ end: 5, start: 0 }]
    normalizeComposerEditorDom(editor)

    expect(texts(editor)).toEqual(['alpha'])
    expect(composerPlainText(editor)).toBe('alpha beta')
  })

  it('clips a wrapper to the range the provider now names', () => {
    let ranges: ComposerNonProseRange[] = [{ end: 10, start: 0 }]

    addProvider('spellcheck', () => ranges)

    const editor = editorFor('0123456789abcdef')

    expect(texts(editor)).toEqual(['0123456789'])

    ranges = [{ end: 6, start: 0 }]
    normalizeComposerEditorDom(editor)

    // Dropped, before: the wrapper covered 0..10 and no range covered it whole.
    expect(texts(editor)).toEqual(['012345'])
    expect(composerPlainText(editor)).toBe('0123456789abcdef')
  })

  it('extends a wrapper over text the provider now covers', () => {
    let ranges: ComposerNonProseRange[] = [{ end: 5, start: 0 }]

    addProvider('spellcheck', () => ranges)

    const editor = editorFor('alpha beta')

    expect(texts(editor)).toEqual(['alpha'])

    ranges = [{ end: 7, start: 0 }]
    normalizeComposerEditorDom(editor)

    expect(texts(editor)).toEqual(['alpha b'])
    expect(composerPlainText(editor)).toBe('alpha beta')
  })

  it('leaves every node alone when the answer still matches the DOM', () => {
    addProvider('spellcheck', ({ text: draft }) => [{ end: draft.length, start: 0 }])

    const editor = editorFor('keep me')
    const [wrapper] = wrappers(editor)
    const text = wrapper.firstChild

    normalizeComposerEditorDom(editor)

    expect(wrappers(editor)).toHaveLength(1)
    expect(wrappers(editor)[0]).toBe(wrapper)
    expect(wrapper.firstChild).toBe(text)
  })
})

describe('composer.nonProse — a wrapper Chromium block-wrapped', () => {
  it('is reconciled at any depth', () => {
    let ranges: ComposerNonProseRange[] = [{ end: 5, start: 0 }]

    addProvider('spellcheck', () => ranges)

    const editor = editorFor('alpha beta')

    // Chromium block-wraps a line while leaving siblings behind.
    const block = document.createElement('div')

    block.append(...Array.from(editor.childNodes))
    editor.append(document.createTextNode(' tail'), block)

    expect(block.querySelector('[data-composer-nonprose]')).not.toBeNull()

    ranges = []
    normalizeComposerEditorDom(editor)

    expect(wrappers(editor)).toHaveLength(0)
    expect(composerPlainText(editor)).toBe(' tailalpha beta\n')
  })
})

describe('composer.nonProse — boundaries and unload', () => {
  it('snaps a range boundary off a surrogate pair', () => {
    // [1, 2) of 'a😀b' cuts the emoji in half. A DOM node cannot hold half a
    // code point, so the wrapper takes the whole pair.
    addProvider('spellcheck', () => [{ end: 2, start: 1 }])

    const editor = editorFor('a😀b')

    expect(texts(editor)).toEqual(['😀'])
    expect(editor.innerHTML).not.toMatch(/[\ud800-\udbff](?![\udc00-\udfff])/)
    expect(composerPlainText(editor)).toBe('a😀b')
  })

  it('unwraps at unload, without waiting for a reconcile', () => {
    addProvider('spellcheck', ({ text: draft }) => [{ end: draft.length, start: 0 }])

    const editor = editorFor('plugin text')

    document.body.append(editor)
    expect(wrappers(editor)).toHaveLength(1)

    // No normalizeComposerEditorDom: an unload schedules no pass of its own.
    disposers.splice(0).forEach(dispose => dispose())

    expect(wrappers(editor)).toHaveLength(0)
    expect(editor.innerHTML).toBe('plugin text')
  })
})

describe('composer.nonProse — one draft, both passes', () => {
  it('resolves the render pass against the string the DOM serializes back to', () => {
    const seen: string[] = []

    addProvider('spellcheck', ({ text }) => {
      seen.push(text)

      return []
    })

    // The source form is unquoted; the chip serializes quoted.
    const editor = editorFor('`a` @file:src/a.ts `b`')
    const plain = composerPlainText(editor)

    expect(editor.querySelector('[data-ref-text]')).not.toBeNull()
    expect(plain).toBe('`a` @file:`src/a.ts` `b`')
    expect(seen.at(-1)).toBe(plain)

    seen.length = 0
    normalizeComposerEditorDom(editor)

    expect(seen).toEqual([plain])
  })

  it('lands a range answered for that string on the right characters', () => {
    // 0..3 is '`a`'; the chip's serialized form starts at 4.
    addProvider('spellcheck', () => [{ end: 3, start: 0 }])

    const editor = editorFor('`a` @file:src/a.ts `b`')

    expect(texts(editor)).toEqual(['`a`'])
    expect(composerPlainText(editor)).toBe('`a` @file:`src/a.ts` `b`')
  })
})

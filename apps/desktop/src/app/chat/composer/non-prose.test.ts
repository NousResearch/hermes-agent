import { afterEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'

import { COMPOSER_AREAS, type ComposerNonProseProvider, type ComposerNonProseRange } from './contrib'
import { COMPOSER_NON_PROSE_PER_PROVIDER, COMPOSER_NON_PROSE_TOTAL } from './non-prose'
import {
  beginComposerComposition,
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
})

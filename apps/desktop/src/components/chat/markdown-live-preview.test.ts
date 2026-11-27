import { markdown } from '@codemirror/lang-markdown'
import { ensureSyntaxTree } from '@codemirror/language'
import { EditorSelection, EditorState } from '@codemirror/state'
import type { Decoration } from '@codemirror/view'
import { describe, expect, it } from 'vitest'

import { buildLivePreviewDecorations } from './markdown-live-preview'

interface Found {
  className?: string
  from: number
  replaced: boolean
  text: string
  to: number
}

function decorate(doc: string, caret?: number): Found[] {
  const state = EditorState.create({
    doc,
    extensions: [markdown()],
    selection: caret === undefined ? undefined : EditorSelection.cursor(caret)
  })

  ensureSyntaxTree(state, state.doc.length, 5_000)
  const set = buildLivePreviewDecorations(state, [{ from: 0, to: state.doc.length }])
  const found: Found[] = []

  set.between(0, state.doc.length, (from, to, deco: Decoration) => {
    found.push({
      className: deco.spec.class as string | undefined,
      from,
      replaced: Boolean(deco.spec.widget) || (deco.spec.class === undefined && from !== to),
      text: state.doc.sliceString(from, to),
      to
    })
  })

  return found
}

const hiddenText = (found: Found[]) => found.filter(f => f.replaced).map(f => f.text)

describe('markdownLivePreview decorations', () => {
  // Caret parked on the last (plain) line so nothing above is "being edited".
  const doc = '# Title\n\nSome **bold** and *it* and `code` and [link](https://x.y).\n\n- item\n\n> quote\n\nend'
  const away = doc.length

  it('styles headings and hides their marks away from the caret', () => {
    const found = decorate(doc, away)

    expect(found.some(f => f.className?.includes('cm-md-h1') && f.from === 0)).toBe(true)
    expect(hiddenText(found)).toContain('# ')
  })

  it('hides inline syntax marks and the link target, keeping the label', () => {
    const hidden = hiddenText(decorate(doc, away))

    expect(hidden).toEqual(expect.arrayContaining(['**', '*', '`', '[', ']', '(', ')', 'https://x.y']))
    expect(hidden).not.toContain('link')
  })

  it('reveals marks of the construct under the caret only', () => {
    const boldStart = doc.indexOf('**bold**')
    const found = decorate(doc, boldStart + 3)
    const hidden = found.filter(f => f.replaced)

    // `**` around "bold" stay visible…
    expect(hidden.some(f => f.from >= boldStart && f.to <= boldStart + 8)).toBe(false)
    // …while the heading line elsewhere is still rendered.
    expect(hidden.some(f => f.text === '# ')).toBe(true)
  })

  it('renders list bullets and quote lines', () => {
    const found = decorate(doc, away)

    expect(found.some(f => f.text === '-' && f.replaced)).toBe(true)
    expect(found.some(f => f.className === 'cm-md-quote')).toBe(true)
    expect(hiddenText(found)).toContain('> ')
  })

  it('leaves fenced code untouched apart from the block styling', () => {
    const code = '```js\nconst a = **b**\n```\n\nend'
    const found = decorate(code, code.length)

    expect(found.filter(f => f.replaced)).toEqual([])
    expect(found.filter(f => f.className === 'cm-md-codeblock')).toHaveLength(3)
  })

  it('never edits the document text', () => {
    const state = EditorState.create({ doc, extensions: [markdown()] })
    ensureSyntaxTree(state, state.doc.length, 5_000)
    buildLivePreviewDecorations(state, [{ from: 0, to: state.doc.length }])

    expect(state.doc.toString()).toBe(doc)
  })
})

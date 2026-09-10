import type { KeyboardEvent } from 'react'
import { describe, expect, it } from 'vitest'

import { composerPlainText, refChipElement, RICH_INPUT_SLOT } from './rich-editor'
import { chipTypedUrlOnSpace, linkifyUrls, markdownLinkFor, resolveExactLinkPaste, selectionLinkLabel } from './url-refs'

/** An editor holding `text` with a collapsed caret at `caret`, plus the space
 *  keydown the composer would hand `chipTypedUrlOnSpace`. */
const spaceOn = (text: string, caret: number) => {
  const editor = document.createElement('div')
  editor.dataset.slot = RICH_INPUT_SLOT
  editor.textContent = text
  document.body.append(editor)

  const selection = window.getSelection()!
  const range = document.createRange()

  range.setStart(editor.firstChild!, caret)
  range.collapse(true)
  selection.removeAllRanges()
  selection.addRange(range)

  return { editor, event: { currentTarget: editor, key: ' ' } as KeyboardEvent<HTMLDivElement> }
}

describe('linkifyUrls', () => {
  it('rewrites a bare link as a url directive', () => {
    expect(linkifyUrls('https://example.dev/a/b')).toBe('@url:`https://example.dev/a/b`')
  })

  it('keeps the link in place mid-sentence and leaves its punctuation behind', () => {
    expect(linkifyUrls('read https://example.dev/a. then stop')).toBe('read @url:`https://example.dev/a`. then stop')
  })

  it('keeps balanced parens but drops the one that closed the sentence', () => {
    expect(linkifyUrls('(see https://en.wikipedia.org/wiki/A_(b))')).toBe(
      '(see @url:`https://en.wikipedia.org/wiki/A_(b)`)'
    )
  })

  it('rewrites every link in a multi-link paste', () => {
    expect(linkifyUrls('http://a.dev and https://b.dev')).toBe('@url:`http://a.dev` and @url:`https://b.dev`')
  })

  it('leaves a link that is already a directive alone', () => {
    expect(linkifyUrls('@url:`https://example.dev`')).toBe('@url:`https://example.dev`')
  })

  it('leaves text without a scheme alone', () => {
    expect(linkifyUrls('example.dev/a and src/foo.ts')).toBe('example.dev/a and src/foo.ts')
  })
})

describe('resolveExactLinkPaste', () => {
  it('accepts a lone bare link', () => {
    expect(resolveExactLinkPaste('https://example.dev/a/b')).toBe('https://example.dev/a/b')
  })

  it('accepts a wrapped <link> and surrounding whitespace', () => {
    expect(resolveExactLinkPaste('  <https://example.dev/a>  ')).toBe('https://example.dev/a')
  })

  it('rejects prose around the link', () => {
    expect(resolveExactLinkPaste('see https://example.dev')).toBeNull()
    expect(resolveExactLinkPaste('https://example.dev is nice')).toBeNull()
  })

  it('rejects multiple links', () => {
    expect(resolveExactLinkPaste('https://a.dev https://b.dev')).toBeNull()
  })

  it('rejects trailing sentence punctuation and hostless schemes', () => {
    expect(resolveExactLinkPaste('https://example.dev.')).toBeNull()
    expect(resolveExactLinkPaste('https://')).toBeNull()
  })
})

describe('selectionLinkLabel', () => {
  const selectAll = (build: (editor: HTMLElement) => void) => {
    const editor = document.createElement('div')
    editor.dataset.slot = RICH_INPUT_SLOT
    build(editor)
    document.body.append(editor)

    const selection = window.getSelection()!
    const range = document.createRange()

    range.selectNodeContents(editor)
    selection.removeAllRanges()
    selection.addRange(range)

    return editor
  }

  it('returns the selected text', () => {
    const editor = selectAll(node => {
      node.textContent = 'the docs'
    })

    expect(selectionLinkLabel(editor)).toBe('the docs')
    editor.remove()
  })

  it('rejects a collapsed selection', () => {
    const editor = document.createElement('div')
    editor.textContent = 'text'
    document.body.append(editor)
    window.getSelection()?.removeAllRanges()

    expect(selectionLinkLabel(editor)).toBeNull()
    editor.remove()
  })

  it('rejects a selection containing a chip', () => {
    const editor = selectAll(node => {
      node.append(document.createTextNode('see '), refChipElement('url', '`https://a.dev`'))
    })

    expect(selectionLinkLabel(editor)).toBeNull()
    editor.remove()
  })

  it('rejects a multi-line selection', () => {
    const editor = selectAll(node => {
      node.append(document.createTextNode('one'), document.createElement('br'), document.createTextNode('two'))
    })

    expect(selectionLinkLabel(editor)).toBeNull()
    editor.remove()
  })
})

describe('markdownLinkFor', () => {
  it('builds a markdown link', () => {
    expect(markdownLinkFor('the docs', 'https://example.dev')).toBe('[the docs](https://example.dev)')
  })

  it('escapes square brackets in the label', () => {
    expect(markdownLinkFor('a [b] c', 'https://example.dev')).toBe('[a \\[b\\] c](https://example.dev)')
  })
})

describe('chipTypedUrlOnSpace', () => {
  it('chips a link typed right before the caret and adds the space', () => {
    const { editor, event } = spaceOn('see https://example.dev/a', 25)

    expect(chipTypedUrlOnSpace(event)).toBe(true)
    expect(composerPlainText(editor)).toBe('see @url:`https://example.dev/a` ')

    editor.remove()
  })

  it('keeps sentence punctuation outside the chip', () => {
    const { editor, event } = spaceOn('https://example.dev.', 20)

    expect(chipTypedUrlOnSpace(event)).toBe(true)
    expect(composerPlainText(editor)).toBe('@url:`https://example.dev`. ')

    editor.remove()
  })

  it('ignores a caret that is not sitting on a link', () => {
    const { editor, event } = spaceOn('https://example.dev is nice', 27)

    expect(chipTypedUrlOnSpace(event)).toBe(false)
    expect(composerPlainText(editor)).toBe('https://example.dev is nice')

    editor.remove()
  })

  it('ignores a scheme with no host yet', () => {
    const { editor, event } = spaceOn('https://', 8)

    expect(chipTypedUrlOnSpace(event)).toBe(false)

    editor.remove()
  })

  it('leaves a modified space alone', () => {
    const { editor, event } = spaceOn('https://example.dev', 19)

    expect(chipTypedUrlOnSpace({ ...event, altKey: true })).toBe(false)
    expect(composerPlainText(editor)).toBe('https://example.dev')

    editor.remove()
  })
})

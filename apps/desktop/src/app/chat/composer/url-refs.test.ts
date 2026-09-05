import type { KeyboardEvent } from 'react'
import { describe, expect, it } from 'vitest'

import { composerPlainText, RICH_INPUT_SLOT } from './rich-editor'
import { chipTypedUrlOnSpace, linkifyUrls } from './url-refs'

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

  it('preserves links inside fenced code while rewriting surrounding prose', () => {
    const text = [
      'before https://before.dev',
      '```ini',
      'endpoint=https://code.dev/api',
      '```',
      'after https://after.dev'
    ].join('\n')

    expect(linkifyUrls(text)).toBe(
      [
        'before @url:`https://before.dev`',
        '```ini',
        'endpoint=https://code.dev/api',
        '```',
        'after @url:`https://after.dev`'
      ].join('\n')
    )
  })

  it('preserves links inside tilde-fenced code', () => {
    const text = ['~~~yaml', 'endpoint: https://code.dev/api', '~~~'].join('\n')

    expect(linkifyUrls(text)).toBe(text)
  })

  it('preserves links inside inline code while rewriting surrounding prose', () => {
    expect(linkifyUrls('see https://before.dev then `curl https://code.dev/api` and https://after.dev')).toBe(
      'see @url:`https://before.dev` then `curl https://code.dev/api` and @url:`https://after.dev`'
    )
  })

  it('supports arbitrary backtick runs around inline code', () => {
    const text = 'run ``curl ` https://code.dev/api`` now'

    expect(linkifyUrls(text)).toBe(text)
  })

  it('preserves links in unfinished code while the user is composing it', () => {
    const unfinishedFence = ['```sh', 'curl https://code.dev/api'].join('\n')
    const unfinishedInline = 'run `curl https://code.dev/api'

    expect(linkifyUrls(unfinishedFence)).toBe(unfinishedFence)
    expect(linkifyUrls(unfinishedInline)).toBe(unfinishedInline)
  })

  it('rewrites prose links after an escaped unmatched backtick', () => {
    expect(linkifyUrls('show \\` literally, then visit https://example.dev/api')).toBe(
      'show \\` literally, then visit @url:`https://example.dev/api`'
    )
  })

  it('leaves text without a scheme alone', () => {
    expect(linkifyUrls('example.dev/a and src/foo.ts')).toBe('example.dev/a and src/foo.ts')
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

  it('does not chip a link typed inside an unfinished fenced code block', () => {
    const text = ['```sh', 'curl https://code.dev/api'].join('\n')
    const { editor, event } = spaceOn(text, text.length)

    expect(chipTypedUrlOnSpace(event)).toBe(false)
    expect(composerPlainText(editor)).toBe(text)

    editor.remove()
  })

  it('does not chip a link typed inside an unfinished inline code span', () => {
    const text = 'run `curl https://code.dev/api'
    const { editor, event } = spaceOn(text, text.length)

    expect(chipTypedUrlOnSpace(event)).toBe(false)
    expect(composerPlainText(editor)).toBe(text)

    editor.remove()
  })

  it('still chips a link typed after a completed code block', () => {
    const text = ['```', 'https://code.dev/api', '```', 'see https://example.dev/api'].join('\n')
    const { editor, event } = spaceOn(text, text.length)

    expect(chipTypedUrlOnSpace(event)).toBe(true)
    expect(composerPlainText(editor)).toBe(
      ['```', 'https://code.dev/api', '```', 'see @url:`https://example.dev/api` '].join('\n')
    )

    editor.remove()
  })

  it('chips a prose link typed after an escaped unmatched backtick', () => {
    const text = 'show \\` literally, then visit https://example.dev/api'
    const { editor, event } = spaceOn(text, text.length)

    expect(chipTypedUrlOnSpace(event)).toBe(true)
    expect(composerPlainText(editor)).toBe('show \\` literally, then visit @url:`https://example.dev/api` ')

    editor.remove()
  })
})

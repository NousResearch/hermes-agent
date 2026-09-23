import type { KeyboardEvent } from 'react'
import { describe, expect, it } from 'vitest'

import { chipTypedPathOnSpace, pathifyRefs } from './path-refs'
import { composerPlainText, RICH_INPUT_SLOT } from './rich-editor'

/**
 * Tab-descending into a folder from the `@` popover leaves a bare
 * `@apps/desktop/` in the editor. That is not a reference — the backend's
 * REFERENCE_PATTERN only matches `@kind:value` — so it used to submit as plain
 * text and silently attach nothing.
 */

/** An editor holding `text` with a collapsed caret at its end, plus the space
 *  keydown the composer would hand `chipTypedPathOnSpace`. */
const spaceOn = (text: string) => {
  const editor = document.createElement('div')
  editor.dataset.slot = RICH_INPUT_SLOT
  editor.textContent = text
  document.body.append(editor)

  const selection = window.getSelection()!
  const range = document.createRange()

  range.setStart(editor.firstChild!, text.length)
  range.collapse(true)
  selection.removeAllRanges()
  selection.addRange(range)

  return { editor, event: { currentTarget: editor, key: ' ' } as KeyboardEvent<HTMLDivElement> }
}

describe('pathifyRefs', () => {
  it('promotes a Tab-descended folder token', () => {
    expect(pathifyRefs('look at @apps/desktop/')).toBe('look at @folder:`apps/desktop`')
  })

  it('promotes a typed file path', () => {
    expect(pathifyRefs('read @src/main.ts please')).toBe('read @file:`src/main.ts` please')
  })

  it('promotes every bare path in one message', () => {
    expect(pathifyRefs('@apps/desktop/ and @src/main.ts')).toBe('@folder:`apps/desktop` and @file:`src/main.ts`')
  })

  it('leaves an already-typed directive alone', () => {
    const text = 'look at @folder:`apps/desktop` ok'

    expect(pathifyRefs(text)).toBe(text)
  })

  it('leaves a handle alone', () => {
    expect(pathifyRefs('cc @teknium1 on this')).toBe('cc @teknium1 on this')
  })

  it('leaves simple refs alone', () => {
    expect(pathifyRefs('check @diff and @staged')).toBe('check @diff and @staged')
  })

  it('leaves an email alone', () => {
    expect(pathifyRefs('mail me at brooklyn@nous.dev/x')).toBe('mail me at brooklyn@nous.dev/x')
  })

  it('is a no-op without an @', () => {
    expect(pathifyRefs('nothing here')).toBe('nothing here')
  })
})

describe('chipTypedPathOnSpace', () => {
  it('chips a Tab-descended folder token into a real @folder: chip', () => {
    const { editor, event } = spaceOn('look at @apps/desktop/')

    expect(chipTypedPathOnSpace(event)).toBe(true)

    const chip = editor.querySelector('[data-ref-text]')

    expect(chip?.getAttribute('data-ref-kind')).toBe('folder')
    expect(chip?.getAttribute('data-ref-id')).toBe('apps/desktop')
    expect(chip?.textContent).toContain('desktop')
    expect(composerPlainText(editor)).toBe('look at @folder:`apps/desktop` ')
  })

  it('chips a typed file path', () => {
    const { editor, event } = spaceOn('read @src/main.ts')

    expect(chipTypedPathOnSpace(event)).toBe(true)
    expect(editor.querySelector('[data-ref-kind="file"]')).not.toBeNull()
    expect(composerPlainText(editor)).toBe('read @file:`src/main.ts` ')
  })

  it('leaves a handle alone', () => {
    const { editor, event } = spaceOn('cc @teknium1')

    expect(chipTypedPathOnSpace(event)).toBe(false)
    expect(editor.querySelector('[data-ref-text]')).toBeNull()
  })

  it('leaves an already-typed ref alone', () => {
    const { event } = spaceOn('look at @folder:`apps/desktop`')

    expect(chipTypedPathOnSpace(event)).toBe(false)
  })
})

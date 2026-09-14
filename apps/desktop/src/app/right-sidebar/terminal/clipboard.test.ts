import { afterEach, describe, expect, it } from 'vitest'

import { mirrorSelection, terminalClipboardIntent } from './clipboard'

afterEach(() => {
  window.getSelection()?.removeAllRanges()
  document.body.replaceChildren()
})

const key = (init: Partial<KeyboardEvent> & { key: string }) =>
  ({ altKey: false, ctrlKey: false, metaKey: false, shiftKey: false, type: 'keydown', ...init }) as KeyboardEvent

describe('terminalClipboardIntent', () => {
  it('never claims a bare Ctrl+C with nothing selected, on either platform', () => {
    for (const isMac of [true, false]) {
      expect(terminalClipboardIntent(key({ ctrlKey: true, key: 'c' }), { hasSelection: false, isMac })).toBeNull()
    }
  })

  it('copies on Ctrl+C when text is selected, so a selection is never lost to SIGINT', () => {
    expect(terminalClipboardIntent(key({ ctrlKey: true, key: 'c' }), { hasSelection: true, isMac: false })).toBe('copy')
  })

  it('reserves plain Ctrl+C for the shell on macOS, where ⌘C is the copy chord', () => {
    expect(terminalClipboardIntent(key({ ctrlKey: true, key: 'c' }), { hasSelection: true, isMac: true })).toBeNull()
    expect(terminalClipboardIntent(key({ key: 'c', metaKey: true }), { hasSelection: true, isMac: true })).toBe('copy')
  })

  it('only claims copy when there is something to copy', () => {
    expect(terminalClipboardIntent(key({ key: 'c', metaKey: true }), { hasSelection: false, isMac: true })).toBeNull()
    expect(
      terminalClipboardIntent(key({ ctrlKey: true, key: 'c', shiftKey: true }), { hasSelection: false, isMac: false })
    ).toBeNull()
  })

  it('claims paste regardless of selection, since paste has nothing to do with one', () => {
    expect(terminalClipboardIntent(key({ key: 'v', metaKey: true }), { hasSelection: false, isMac: true })).toBe(
      'paste'
    )
    expect(
      terminalClipboardIntent(key({ ctrlKey: true, key: 'v', shiftKey: true }), { hasSelection: false, isMac: false })
    ).toBe('paste')
  })

  it('leaves shell chords alone: bare Ctrl+V, Alt combos, and keyup', () => {
    expect(terminalClipboardIntent(key({ ctrlKey: true, key: 'v' }), { hasSelection: false, isMac: false })).toBeNull()
    expect(
      terminalClipboardIntent(key({ altKey: true, ctrlKey: true, key: 'c' }), { hasSelection: true, isMac: false })
    ).toBeNull()
    expect(
      terminalClipboardIntent(key({ key: 'c', metaKey: true, type: 'keyup' }), { hasSelection: true, isMac: true })
    ).toBeNull()
  })
})

describe('mirrorSelection', () => {
  const host = () => {
    const el = document.createElement('div')
    const textarea = document.createElement('textarea')
    textarea.className = 'xterm-helper-textarea'
    el.appendChild(textarea)
    document.body.appendChild(el)

    return { el, textarea }
  }

  it('puts the selection where the OS copy command can find it while the terminal is focused', () => {
    const { el, textarea } = host()
    textarea.focus()
    mirrorSelection(el, 'npm run check')

    expect(textarea.value).toBe('npm run check')
    expect(textarea.selectionStart).toBe(0)
    expect(textarea.selectionEnd).toBe('npm run check'.length)
  })

  it('keeps the text staged but does not steal the document selection when chat owns focus', () => {
    const { el, textarea } = host()
    const outside = document.createElement('textarea')
    document.body.appendChild(outside)
    outside.focus()

    mirrorSelection(el, 'stale terminal scrap')

    expect(textarea.value).toBe('stale terminal scrap')
    // No `select()` — caret may sit at the end after the value write, but the
    // range must stay collapsed so the OS copy command still sees chat text.
    expect(textarea.selectionStart).toBe(textarea.selectionEnd)
    expect(document.activeElement).toBe(outside)
  })

  it('does not clobber a live chat highlight even if the terminal still has focus', () => {
    const { el, textarea } = host()
    textarea.focus()

    const outside = document.createElement('span')
    outside.textContent = 'chat text'
    document.body.appendChild(outside)
    const range = document.createRange()
    range.selectNodeContents(outside)
    const selection = window.getSelection()!
    selection.removeAllRanges()
    selection.addRange(range)

    mirrorSelection(el, 'terminal scrap')

    expect(textarea.value).toBe('terminal scrap')
    expect(textarea.selectionStart).toBe(textarea.selectionEnd)
    expect(selection.toString()).toBe('chat text')
  })

  it('clears the mirror when the selection goes away', () => {
    const { el, textarea } = host()
    textarea.focus()
    mirrorSelection(el, 'something')
    mirrorSelection(el, '')

    expect(textarea.value).toBe('')
  })

  it('is a no-op before xterm has mounted its textarea', () => {
    expect(() => mirrorSelection(document.createElement('div'), 'text')).not.toThrow()
  })
})

import { describe, expect, it } from 'vitest'

import { mirrorSelection, terminalClipboardIntent } from './clipboard'

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
    expect(terminalClipboardIntent(key({ key: 'v', metaKey: true }), { hasSelection: false, isMac: true })).toBe('paste')
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

    return { el, textarea }
  }

  it('puts the selection where the OS copy command can find it', () => {
    const { el, textarea } = host()
    mirrorSelection(el, 'npm run check')

    expect(textarea.value).toBe('npm run check')
  })

  it('clears the mirror when the selection goes away', () => {
    const { el, textarea } = host()
    mirrorSelection(el, 'something')
    mirrorSelection(el, '')

    expect(textarea.value).toBe('')
  })

  it('is a no-op before xterm has mounted its textarea', () => {
    expect(() => mirrorSelection(document.createElement('div'), 'text')).not.toThrow()
  })
})

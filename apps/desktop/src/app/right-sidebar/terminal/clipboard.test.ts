import { afterEach, describe, expect, it } from 'vitest'

import { mirrorSelection, terminalClipboardIntent } from './clipboard'

afterEach(() => {
  window.getSelection()?.removeAllRanges()
  document.body.replaceChildren()
})

const key = (init: Partial<KeyboardEvent> & { key: string }) =>
  ({ altKey: false, ctrlKey: false, metaKey: false, shiftKey: false, type: 'keydown', ...init }) as KeyboardEvent

describe('terminalClipboardIntent', () => {
  it('never claims a bare Ctrl+C with nothing selected, on any platform', () => {
    for (const [isMac, isWindows] of [
      [true, false],
      [false, true],
      [false, false]
    ]) {
      expect(
        terminalClipboardIntent(key({ ctrlKey: true, key: 'c' }), { hasSelection: false, isMac, isWindows })
      ).toBeNull()
    }
  })

  it('copies on Ctrl+C when text is selected, so a selection is never lost to SIGINT', () => {
    expect(
      terminalClipboardIntent(key({ ctrlKey: true, key: 'c' }), { hasSelection: true, isMac: false, isWindows: false })
    ).toBe('copy')
  })

  it('reserves plain Ctrl+C for the shell on macOS, where ⌘C is the copy chord', () => {
    expect(
      terminalClipboardIntent(key({ ctrlKey: true, key: 'c' }), { hasSelection: true, isMac: true, isWindows: false })
    ).toBeNull()
    expect(
      terminalClipboardIntent(key({ key: 'c', metaKey: true }), { hasSelection: true, isMac: true, isWindows: false })
    ).toBe('copy')
  })

  it('only claims copy when there is something to copy', () => {
    expect(
      terminalClipboardIntent(key({ key: 'c', metaKey: true }), { hasSelection: false, isMac: true, isWindows: false })
    ).toBeNull()
    expect(
      terminalClipboardIntent(key({ ctrlKey: true, key: 'c', shiftKey: true }), {
        hasSelection: false,
        isMac: false,
        isWindows: false
      })
    ).toBeNull()
  })

  it('claims paste regardless of selection, since paste has nothing to do with one', () => {
    expect(
      terminalClipboardIntent(key({ key: 'v', metaKey: true }), { hasSelection: false, isMac: true, isWindows: false })
    ).toBe('paste')
    expect(
      terminalClipboardIntent(key({ ctrlKey: true, key: 'v', shiftKey: true }), {
        hasSelection: false,
        isMac: false,
        isWindows: false
      })
    ).toBe('paste')
  })

  // Windows has no application menu (main.ts installs one on macOS only), so
  // Chromium never turns Ctrl+V or Shift+Insert into a Paste editor command and
  // xterm's helper-textarea `paste` listener never fires. Whatever this map
  // declines on Windows is a dead key, which is why these two are claimed.
  it('pastes on bare Ctrl+V on Windows, where VS Code binds it as the primary chord', () => {
    expect(
      terminalClipboardIntent(key({ ctrlKey: true, key: 'v' }), { hasSelection: false, isMac: false, isWindows: true })
    ).toBe('paste')
  })

  it('pastes on Shift+Insert on every platform, since no shell binds it', () => {
    for (const [isMac, isWindows] of [
      [true, false],
      [false, true],
      [false, false]
    ]) {
      expect(
        terminalClipboardIntent(key({ key: 'Insert', shiftKey: true }), { hasSelection: false, isMac, isWindows })
      ).toBe('paste')
    }
  })

  it('leaves a bare or Ctrl-modified Insert to the shell — only Shift+Insert is paste', () => {
    expect(
      terminalClipboardIntent(key({ key: 'Insert' }), { hasSelection: false, isMac: false, isWindows: true })
    ).toBeNull()
    expect(
      terminalClipboardIntent(key({ ctrlKey: true, key: 'Insert', shiftKey: true }), {
        hasSelection: false,
        isMac: false,
        isWindows: true
      })
    ).toBeNull()
    expect(
      terminalClipboardIntent(key({ altKey: true, key: 'Insert', shiftKey: true }), {
        hasSelection: false,
        isMac: false,
        isWindows: true
      })
    ).toBeNull()
  })

  it('leaves shell chords alone: bare Ctrl+V off Windows, Alt combos, and keyup', () => {
    expect(
      terminalClipboardIntent(key({ ctrlKey: true, key: 'v' }), {
        hasSelection: false,
        isMac: false,
        isWindows: false
      })
    ).toBeNull()
    expect(
      terminalClipboardIntent(key({ altKey: true, ctrlKey: true, key: 'c' }), {
        hasSelection: true,
        isMac: false,
        isWindows: false
      })
    ).toBeNull()
    expect(
      terminalClipboardIntent(key({ key: 'c', metaKey: true, type: 'keyup' }), {
        hasSelection: true,
        isMac: true,
        isWindows: false
      })
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

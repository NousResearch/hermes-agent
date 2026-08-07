import { afterEach, describe, expect, it } from 'vitest'

import {
  lineSelectionFromHostRange,
  lineSelectionFromOffsets,
  lineSelectionFromSelectedText,
  offsetOfLineStart,
  preferOffsetFromRange,
  PREVIEW_ADD_TO_CHAT_ATTR,
  previewOwnsAddSelectionShortcut,
  previewSelectionFileLabel,
  readHostTextSelection,
  retainPreviewAddShortcutClaim,
  selectionBelongsToPreviewAddToChat,
  sourceLineSelectionRef
} from './preview-add-to-chat'

afterEach(() => {
  window.getSelection()?.removeAllRanges()
  document.body.replaceChildren()
})

describe('sourceLineSelectionRef', () => {
  it('builds a single-line @line ref relative to cwd', () => {
    expect(sourceLineSelectionRef('/repo/src/app.ts', { end: 12, start: 12 }, '/repo')).toBe('@line:src/app.ts:12')
  })

  it('builds an inclusive range when end is past start', () => {
    expect(sourceLineSelectionRef('/repo/a.py', { end: 20, start: 10 }, '/repo')).toBe('@line:a.py:10-20')
  })

  it('quotes values that contain spaces', () => {
    expect(sourceLineSelectionRef('/repo/my file.ts', { end: 3, start: 3 }, '/repo')).toBe('@line:`my file.ts:3`')
  })

  it('keeps Windows absolute preview paths relative to cwd in the chip ref', () => {
    expect(
      sourceLineSelectionRef('C:\\workspace\\hermes-agent\\apps\\desktop\\foo.ts', { end: 12, start: 12 }, 'C:\\workspace\\hermes-agent')
    ).toBe('@line:apps/desktop/foo.ts:12')
  })
})

describe('previewSelectionFileLabel', () => {
  it('returns the basename for posix and windows paths', () => {
    expect(previewSelectionFileLabel('/tmp/notes.md')).toBe('notes.md')
    expect(previewSelectionFileLabel('C:\\work\\notes.md')).toBe('notes.md')
  })
})

describe('lineSelectionFromOffsets', () => {
  it('maps character offsets to 1-based inclusive lines', () => {
    const text = 'one\ntwo\nthree\n'

    expect(lineSelectionFromOffsets(text, 0, 3)).toEqual({ end: 1, start: 1 })
    expect(lineSelectionFromOffsets(text, 4, 7)).toEqual({ end: 2, start: 2 })
    expect(lineSelectionFromOffsets(text, 0, 11)).toEqual({ end: 3, start: 1 })
  })
})

describe('lineSelectionFromSelectedText', () => {
  it('finds the line span for a unique selection', () => {
    const text = 'alpha\nbeta\ngamma\n'

    expect(lineSelectionFromSelectedText(text, 'beta')).toEqual({ end: 2, start: 2 })
    expect(lineSelectionFromSelectedText(text, 'alpha\nbeta')).toEqual({ end: 2, start: 1 })
  })

  it('prefers the occurrence nearest preferOffset when duplicated', () => {
    const text = 'x\nrepeat\ny\nrepeat\nz\n'

    expect(lineSelectionFromSelectedText(text, 'repeat', 0)).toEqual({ end: 2, start: 2 })
    expect(lineSelectionFromSelectedText(text, 'repeat', 12)).toEqual({ end: 4, start: 4 })
  })
})

describe('previewOwnsAddSelectionShortcut', () => {
  it('tracks nested retain/release so the terminal can defer Cmd/Ctrl+L', () => {
    expect(previewOwnsAddSelectionShortcut()).toBe(false)

    const releaseA = retainPreviewAddShortcutClaim()
    expect(previewOwnsAddSelectionShortcut()).toBe(true)

    const releaseB = retainPreviewAddShortcutClaim()
    releaseA()
    expect(previewOwnsAddSelectionShortcut()).toBe(true)

    releaseB()
    expect(previewOwnsAddSelectionShortcut()).toBe(false)
  })

  it('treats a live selection inside the preview frame as ownership without a claim', () => {
    const host = document.createElement('div')
    host.setAttribute(PREVIEW_ADD_TO_CHAT_ATTR, '')
    const span = document.createElement('span')
    span.textContent = 'select-all me'
    host.appendChild(span)
    document.body.appendChild(host)

    const range = document.createRange()
    range.selectNodeContents(span)
    const selection = window.getSelection()!
    selection.removeAllRanges()
    selection.addRange(range)

    expect(selectionBelongsToPreviewAddToChat()).toBe(true)
    expect(previewOwnsAddSelectionShortcut()).toBe(true)
  })
})

describe('preferOffsetFromRange', () => {
  it('maps nearby data-preview-line markers to a source char offset', () => {
    const source = 'a\nrepeat\nb\nrepeat\nc\n'
    const host = document.createElement('div')
    const gutterEarly = document.createElement('div')
    gutterEarly.dataset.previewLine = '2'
    Object.defineProperty(gutterEarly, 'getBoundingClientRect', {
      value: () => ({ top: 0, bottom: 10, height: 10, left: 0, right: 10, width: 10, x: 0, y: 0, toJSON: () => ({}) })
    })
    const gutterLate = document.createElement('div')
    gutterLate.dataset.previewLine = '4'
    Object.defineProperty(gutterLate, 'getBoundingClientRect', {
      value: () => ({ top: 40, bottom: 50, height: 10, left: 0, right: 10, width: 10, x: 0, y: 40, toJSON: () => ({}) })
    })
    const span = document.createElement('span')
    span.textContent = 'repeat'
    host.append(gutterEarly, gutterLate, span)
    document.body.appendChild(host)

    const range = document.createRange()
    range.selectNodeContents(span)
    Object.defineProperty(range, 'getBoundingClientRect', {
      value: () => ({ top: 42, bottom: 48, height: 6, left: 20, right: 80, width: 60, x: 20, y: 42, toJSON: () => ({}) })
    })

    expect(preferOffsetFromRange(source, host, range)).toBe(offsetOfLineStart(source, 4))
    expect(lineSelectionFromSelectedText(source, 'repeat', preferOffsetFromRange(source, host, range))).toEqual({
      end: 4,
      start: 4
    })
  })
})

describe('readHostTextSelection', () => {
  it('returns null when nothing is selected inside the host', () => {
    const host = document.createElement('div')
    document.body.appendChild(host)

    expect(readHostTextSelection(host)).toBeNull()
  })

  it('reads a live selection that belongs to the host', () => {
    const host = document.createElement('div')
    const span = document.createElement('span')
    span.textContent = 'quote me please'
    host.appendChild(span)
    document.body.appendChild(host)

    const range = document.createRange()
    range.selectNodeContents(span)
    const selection = window.getSelection()!
    selection.removeAllRanges()
    selection.addRange(range)

    expect(readHostTextSelection(host)?.text).toBe('quote me please')
  })

  it('ignores selections outside the host', () => {
    const host = document.createElement('div')
    const outsider = document.createElement('span')
    outsider.textContent = 'outside'
    document.body.append(host, outsider)

    const range = document.createRange()
    range.selectNodeContents(outsider)
    const selection = window.getSelection()!
    selection.removeAllRanges()
    selection.addRange(range)

    expect(readHostTextSelection(host)).toBeNull()
  })

  it('keeps a drag that starts in the host but ends outside (commonAncestor leaves the frame)', () => {
    const host = document.createElement('div')
    const span = document.createElement('span')
    span.textContent = 'inside selection'
    host.appendChild(span)
    const outsider = document.createElement('span')
    outsider.textContent = ' and outside'
    document.body.append(host, outsider)

    const range = document.createRange()
    range.setStart(span.firstChild!, 0)
    range.setEnd(outsider.firstChild!, outsider.textContent!.length)
    const selection = window.getSelection()!
    selection.removeAllRanges()
    selection.addRange(range)

    // commonAncestor is body — the old contains(commonAncestor) check dropped these.
    expect(host.contains(range.commonAncestorContainer)).toBe(false)
    expect(readHostTextSelection(host)?.text).toBe('inside selection')
    expect(selectionBelongsToPreviewAddToChat()).toBe(false)

    host.setAttribute(PREVIEW_ADD_TO_CHAT_ATTR, '')
    expect(selectionBelongsToPreviewAddToChat()).toBe(true)
  })
})

describe('lineSelectionFromHostRange', () => {
  it('falls back to gutter markers when DOM text does not match source', () => {
    const source = 'alpha\nbeta\ngamma\n'
    const host = document.createElement('div')
    const gutter = document.createElement('div')
    gutter.dataset.previewLine = '2'
    Object.defineProperty(gutter, 'getBoundingClientRect', {
      value: () => ({ top: 20, bottom: 30, height: 10, left: 0, right: 10, width: 10, x: 0, y: 20, toJSON: () => ({}) })
    })
    // Simulated virtualized / highlighted DOM that dropped the newline between lines.
    const span = document.createElement('span')
    span.textContent = 'alphabeta'
    host.append(gutter, span)
    document.body.appendChild(host)

    const range = document.createRange()
    range.selectNodeContents(span)
    Object.defineProperty(range, 'getBoundingClientRect', {
      value: () => ({ top: 22, bottom: 28, height: 6, left: 20, right: 80, width: 60, x: 20, y: 22, toJSON: () => ({}) })
    })

    expect(lineSelectionFromSelectedText(source, range.toString())).toBeNull()
    expect(lineSelectionFromHostRange(source, host, range)).toEqual({ end: 2, start: 2 })
  })
})

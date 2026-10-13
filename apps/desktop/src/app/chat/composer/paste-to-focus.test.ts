import { afterEach, describe, expect, it, vi } from 'vitest'

import { onComposerAttachImagesRequest, onComposerFocusRequest, onComposerInsertRequest } from './focus'
import { handleWindowPaste, routeClipboardToComposer } from './paste-to-focus'

/** Minimal DataTransfer stand-in: text/plain + optional image file items. */
function clipboard({ files = [] as File[], text = '' } = {}): DataTransfer {
  return {
    files: { item: (i: number) => files[i] ?? null, length: files.length },
    getData: (type: string) => (type === 'text' || type === 'text/plain' ? text : ''),
    items: files.map(file => ({ getAsFile: () => file, kind: 'file', type: file.type }))
  } as unknown as DataTransfer
}

const image = (name = 'shot.png') => new File([new Uint8Array(8192)], name, { type: 'image/png' })

/** The bus defers dispatch a macrotask; flush it. */
const flushBus = () => new Promise(resolve => setTimeout(resolve, 1))

function pasteEvent(clip: DataTransfer, target: EventTarget = document.body): ClipboardEvent {
  const event = new Event('paste', { bubbles: true, cancelable: true }) as ClipboardEvent

  Object.defineProperty(event, 'clipboardData', { value: clip })
  Object.defineProperty(event, 'target', { value: target })

  return event
}

afterEach(() => {
  document.body.replaceChildren()
  vi.restoreAllMocks()
})

describe('routeClipboardToComposer', () => {
  it('routes pasted text to the composer as an insert', async () => {
    const inserts: string[] = []
    const off = onComposerInsertRequest(({ text }) => inserts.push(text))

    expect(routeClipboardToComposer(clipboard({ text: 'hello there' }))).toBe(true)
    await flushBus()
    off()

    expect(inserts).toEqual(['hello there'])
  })

  it('chips a pasted link the same way a focused paste does', async () => {
    const inserts: string[] = []
    const off = onComposerInsertRequest(({ text }) => inserts.push(text))

    routeClipboardToComposer(clipboard({ text: 'see https://example.com/docs' }))
    await flushBus()
    off()

    expect(inserts[0]).toContain('@url:')
  })

  it('attaches clipboard images and pulls focus on an image-only paste', async () => {
    const attached: Blob[][] = []
    const focused: boolean[] = []
    const offAttach = onComposerAttachImagesRequest(({ blobs }) => attached.push(blobs))
    const offFocus = onComposerFocusRequest(() => focused.push(true))

    expect(routeClipboardToComposer(clipboard({ files: [image()] }))).toBe(true)
    await flushBus()
    offAttach()
    offFocus()

    expect(attached).toHaveLength(1)
    expect(attached[0]).toHaveLength(1)
    expect(focused).toHaveLength(1)
  })

  it('takes both from a mixed paste — images attach AND the text inserts', async () => {
    const attached: Blob[][] = []
    const inserts: string[] = []
    const offAttach = onComposerAttachImagesRequest(({ blobs }) => attached.push(blobs))
    const offInsert = onComposerInsertRequest(({ text }) => inserts.push(text))

    routeClipboardToComposer(clipboard({ files: [image()], text: 'look at this' }))
    await flushBus()
    offAttach()
    offInsert()

    expect(attached).toHaveLength(1)
    expect(inserts).toEqual(['look at this'])
  })

  it('reports an empty clipboard as unhandled', () => {
    expect(routeClipboardToComposer(clipboard())).toBe(false)
  })
})

describe('handleWindowPaste', () => {
  it('swallows a routed paste on non-editable chrome', () => {
    const event = pasteEvent(clipboard({ text: 'hi' }))

    handleWindowPaste(event)

    expect(event.defaultPrevented).toBe(true)
  })

  it('yields to editable targets — they own their own paste', () => {
    const input = document.createElement('input')
    document.body.append(input)
    const event = pasteEvent(clipboard({ text: 'hi' }), input)

    handleWindowPaste(event)

    expect(event.defaultPrevented).toBe(false)
  })

  it('yields while a dialog covers the chat, like type-to-focus', () => {
    const dialog = document.createElement('div')
    dialog.setAttribute('role', 'dialog')
    document.body.append(dialog)
    const event = pasteEvent(clipboard({ text: 'hi' }))

    handleWindowPaste(event)

    expect(event.defaultPrevented).toBe(false)
  })

  it('leaves an already-handled or empty paste alone', () => {
    const handled = pasteEvent(clipboard({ text: 'hi' }))
    handled.preventDefault()
    const before = handled.defaultPrevented

    handleWindowPaste(handled)
    expect(before).toBe(true)

    const empty = pasteEvent(clipboard())
    handleWindowPaste(empty)
    expect(empty.defaultPrevented).toBe(false)
  })
})

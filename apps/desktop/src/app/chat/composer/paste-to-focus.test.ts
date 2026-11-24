import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  ackComposerAttachPastedText,
  onComposerAttachImagesRequest,
  onComposerAttachPastedTextRequest,
  onComposerFocusRequest,
  onComposerInsertRequest,
  requestComposerAttachPastedText
} from './focus'
import { LARGE_PASTE_ATTACHMENT_THRESHOLD } from './large-paste'
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

// The shared large-paste policy. A paste past `LARGE_PASTE_ATTACHMENT_THRESHOLD`
// becomes a `.txt` attachment chip on BOTH paste paths; the unfocused one used
// to insert the raw payload regardless of size, so the two paths disagreed.
describe('large paste on non-editable chrome', () => {
  // The real payload from the 01:19 incident: preamble + final of one
  // assistant bubble, joined with no separator — 1,779 chars.
  const INCIDENT_REPLY = `${'The queue copy lives in localStorage under a different key than I grepped before. Reading it:'}${'x'.repeat(1_686)}`
  // Same shape, past the threshold: what the policy is actually for.
  const HUGE = 'z'.repeat(LARGE_PASTE_ATTACHMENT_THRESHOLD + 1)

  const attachHandler = (result: boolean, received: string[]) =>
    onComposerAttachPastedTextRequest(({ text, token }) => {
      received.push(text)
      ackComposerAttachPastedText(token, result)
    })

  it('turns a paste past the threshold into an attachment instead of composer text', async () => {
    const received: string[] = []
    const offAttach = attachHandler(true, received)
    const inserts: string[] = []
    const offInsert = onComposerInsertRequest(({ text }) => inserts.push(text))

    expect(routeClipboardToComposer(clipboard({ text: HUGE }))).toBe(true)
    await flushBus()
    offAttach()
    offInsert()

    expect(received).toEqual([HUGE])
    // The whole point: none of the payload reached the composer as text.
    expect(inserts).toEqual([])
  })

  it('keeps a paste at or under the threshold inline', async () => {
    const received: string[] = []
    const offAttach = attachHandler(true, received)
    const inserts: string[] = []
    const offInsert = onComposerInsertRequest(({ text }) => inserts.push(text))

    const atThreshold = 'y'.repeat(LARGE_PASTE_ATTACHMENT_THRESHOLD)
    routeClipboardToComposer(clipboard({ text: atThreshold }))
    await flushBus()
    offAttach()
    offInsert()

    expect(received).toEqual([])
    expect(inserts).toEqual([atThreshold])
  })

  it('documents the incident payload: 1,779 chars stays UNDER the threshold, so it still lands inline', async () => {
    // Honest pin on the limit of this fix: the 01:19 payload was 1,779 chars,
    // below LARGE_PASTE_ATTACHMENT_THRESHOLD. Threshold parity makes the two
    // paste paths agree — it does NOT stop a 1.8 KB paste from becoming text.
    expect(INCIDENT_REPLY.length).toBe(1_779)
    expect(INCIDENT_REPLY.length).toBeGreaterThan(LARGE_PASTE_ATTACHMENT_THRESHOLD - 1_500)
    expect(INCIDENT_REPLY.length).toBeLessThan(LARGE_PASTE_ATTACHMENT_THRESHOLD)

    const received: string[] = []
    const offAttach = attachHandler(true, received)
    const inserts: string[] = []
    const offInsert = onComposerInsertRequest(({ text }) => inserts.push(text))

    routeClipboardToComposer(clipboard({ text: INCIDENT_REPLY }))
    await flushBus()
    offAttach()
    offInsert()

    expect(received).toEqual([])
    expect(inserts).toEqual([INCIDENT_REPLY])
  })

  it('falls back to an inline insert when no composer takes the attachment', async () => {
    // No bridge / messaging chat / write failure: the paste must not vanish.
    const offAttach = attachHandler(false, [])
    const inserts: string[] = []
    const offInsert = onComposerInsertRequest(({ text }) => inserts.push(text))

    routeClipboardToComposer(clipboard({ text: HUGE }))
    // Two hops: the request is dispatched on a macrotask, and the fallback
    // insert is itself another dispatched macrotask.
    await flushBus()
    await flushBus()
    offAttach()
    offInsert()

    expect(inserts).toEqual([HUGE])
  })

  it('settles false on the timeout when no composer acks, and a late ack cannot resurrect it', async () => {
    // The request's own contract, not the routed handler's: a composer that never
    // answers must not strand the caller, and an ack arriving after the timeout
    // must not settle an already-settled promise a second time (which would
    // double-insert the payload).
    vi.useFakeTimers()
    const acks: Array<(ok: boolean) => void> = []
    const offAttach = onComposerAttachPastedTextRequest(({ token }) => {
      // Hold the ack: this simulates a composer that mounted but never answered
      // before the timeout (or answered very late).
      acks.push(ok => ackComposerAttachPastedText(token, ok))
    })
    try {
      const pending = requestComposerAttachPastedText(HUGE)
      // No claimant answer: the 5s reply timeout must settle it false.
      await vi.advanceTimersByTimeAsync(6_000)
      await expect(pending).resolves.toBe(false)

      // A late ack, after the timeout, is ignored — the promise is already settled.
      // (`flushBus` uses a real timer, so drain microtasks instead while faked.)
      acks.forEach(ack => ack(true))
      await Promise.resolve()
      await expect(pending).resolves.toBe(false)
    } finally {
      offAttach()
      vi.useRealTimers()
    }
  })

  it('never applies the policy to images — an image-only paste still just attaches', async () => {
    const received: string[] = []
    const offAttach = attachHandler(true, received)
    const attached: Blob[][] = []
    const offImages = onComposerAttachImagesRequest(({ blobs }) => attached.push(blobs))

    routeClipboardToComposer(clipboard({ files: [image()] }))
    await flushBus()
    offAttach()
    offImages()

    expect(received).toEqual([])
    expect(attached).toHaveLength(1)
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

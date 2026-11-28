import assert from 'node:assert/strict'

import { JSDOM } from 'jsdom'
import { describe, test, vi } from 'vitest'

import { installGuestTextSelection, PREVIEW_TEXT_SELECTION_CHANNEL } from './preview-text-selection'

function rig() {
  const dom = new JSDOM('<p>  A precise\nselection — 文本  </p><input value="private"><p data-secret>secret</p>', {
    url: 'http://localhost:3000/preview'
  })

  const doc = dom.window.document
  const listeners = new Map<string, EventListener>()
  vi.spyOn(doc, 'addEventListener').mockImplementation((type, listener) => {
    listeners.set(type, listener as EventListener)
  })
  const sent: unknown[] = []
  let mode: (enabled: unknown) => void = () => undefined

  const dispose = installGuestTextSelection(doc, {
    onMode: listener => {
      mode = listener

      return () => {
        mode = () => undefined
      }
    },
    sendToHost: (channel, payload) => {
      assert.equal(channel, PREVIEW_TEXT_SELECTION_CHANNEL)
      sent.push(payload)
    }
  })

  const select = (node: Node = doc.querySelector('p')!) => {
    const range = doc.createRange()
    range.selectNodeContents(node)
    range.getBoundingClientRect = () => ({ x: 20, y: 30, width: 120, height: 22 }) as DOMRect
    doc.getSelection()!.removeAllRanges()
    doc.getSelection()!.addRange(range)
  }

  const gesture = (type: string, fields: Record<string, unknown> = {}) => {
    listeners.get(type)?.({ isTrusted: true, button: 0, target: doc.querySelector('p'), ...fields } as unknown as Event)
  }

  return { doc, dispose, gesture, mode: (enabled: unknown) => mode(enabled), select, sent }
}

describe('preview text-selection guest bridge', () => {
  test('is opt-in, preserves exact text and accepts only genuine completed selection gestures', () => {
    const r = rig()
    r.select()
    r.gesture('mouseup')
    assert.deepEqual(r.sent, [])
    r.mode(true)
    r.gesture('mouseup', { isTrusted: false })
    r.gesture('mouseup', { button: 2 })
    assert.deepEqual(r.sent, [])
    r.gesture('mouseup')
    assert.deepEqual(r.sent, [
      {
        text: '  A precise\nselection — 文本  ',
        url: 'http://localhost:3000/preview',
        viewportWidth: 1024,
        rect: { x: 20, y: 30, width: 120, height: 22 }
      }
    ])
    r.gesture('keydown', { key: 'Escape' })
    assert.equal(r.sent.at(-1), null)
    r.gesture('keyup', { key: 'ArrowRight', shiftKey: true })
    assert.equal(r.sent.length, 3)
    r.mode(false)
    assert.equal(r.sent.at(-1), null)
    r.gesture('mouseup')
    assert.equal(r.sent.length, 4)
    r.dispose()
  })

  test('rejects editable, secret, excessive and empty selections and dismisses on the next page gesture', () => {
    const r = rig()
    r.mode(true)
    r.select()
    r.gesture('mouseup')
    r.gesture('mousedown')
    assert.equal(r.sent.at(-1), null)
    r.select(r.doc.querySelector('[data-secret]')!)
    r.gesture('mouseup')
    r.select(r.doc.body)
    r.gesture('mouseup')
    r.doc.body.innerHTML = '<div contenteditable="true"><span>editable text</span></div>'
    r.select(r.doc.querySelector('span')!)
    r.gesture('mouseup')
    r.doc.body.textContent = 'x'.repeat(32_769)
    r.select(r.doc.body)
    r.gesture('mouseup')
    r.doc.getSelection()!.removeAllRanges()
    r.gesture('mouseup')
    assert.equal(r.sent.length, 2)
    r.dispose()
  })

  test('does not send geometry from an unavailable viewport', () => {
    const r = rig()
    r.mode(true)
    r.select()

    for (const width of [0, -1, Infinity, NaN]) {
      Object.defineProperty(r.doc.defaultView, 'innerWidth', { configurable: true, value: width })
      r.gesture('mouseup')
    }

    assert.deepEqual(r.sent, [])
    r.dispose()
  })
})

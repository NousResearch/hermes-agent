import { describe, expect, it } from 'vitest'

import {
  addAnnotatePin,
  beginAnnotateMode,
  clearAnnotateStack,
  clearAnnotatePins,
  emptyAnnotateSession,
  emptyAnnotateStack,
  endAnnotateMode,
  removeAnnotatePin,
  updateAnnotatePinNote,
  type AnnotatePinDraft
} from './stack'

const png = 'data:image/png;base64,AAAA'

function draft(note: string, kind: 'area' | 'element' = 'element'): AnnotatePinDraft {
  return {
    imageDataUrl: png,
    kind,
    note,
    pageTitle: 'Demo',
    pageUrl: 'http://localhost:5174/',
    rect: { height: 24, width: 80, x: 10, y: 12 },
    identity:
      kind === 'element'
        ? { css: { 'font-size': '14px' }, selector: 'button.go', tag: 'button', text: 'Go' }
        : undefined
  }
}

describe('annotate pin stack', () => {
  it('numbers click and area pins 1..N as they accumulate', () => {
    let stack = emptyAnnotateStack()
    stack = addAnnotatePin(stack, draft('fix overflow'))
    stack = addAnnotatePin(stack, draft('align this', 'area'))

    expect(stack.pins.map(pin => pin.number)).toEqual([1, 2])
    expect(stack.pins[0]?.kind).toBe('element')
    expect(stack.pins[1]?.kind).toBe('area')
    expect(stack.nextNumber).toBe(3)
  })

  it('keeps numbers stable when a pin is removed', () => {
    let stack = emptyAnnotateStack()
    stack = addAnnotatePin(stack, draft('one'))
    stack = addAnnotatePin(stack, draft('two'))
    stack = addAnnotatePin(stack, draft('three'))
    stack = removeAnnotatePin(stack, stack.pins[1]!.id)
    stack = addAnnotatePin(stack, draft('four'))

    expect(stack.pins.map(pin => pin.number)).toEqual([1, 3, 4])
  })

  it('updates a note without sending anything', () => {
    let stack = emptyAnnotateStack()
    stack = addAnnotatePin(stack, draft(''))
    stack = updateAnnotatePinNote(stack, stack.pins[0]!.id, 'make the label wrap')

    expect(stack.pins[0]?.note).toBe('make the label wrap')
  })

  it('clear resets numbering', () => {
    let stack = addAnnotatePin(emptyAnnotateStack(), draft('x'))
    stack = clearAnnotateStack(stack)
    stack = addAnnotatePin(stack, draft('fresh'))

    expect(stack.pins[0]?.number).toBe(1)
  })

  it('clearing sent pins preserves numbering for the next comment batch', () => {
    let stack = emptyAnnotateStack()
    stack = addAnnotatePin(stack, draft('one'))
    stack = addAnnotatePin(stack, draft('two'))
    stack = clearAnnotatePins(stack)
    stack = addAnnotatePin(stack, draft('three'))

    expect(stack.pins.map(pin => pin.number)).toEqual([3])
  })
})

describe('annotate session mode', () => {
  it('ending mode tears overlay state down and does not auto-send', () => {
    let session = beginAnnotateMode(emptyAnnotateSession())
    session = {
      ...session,
      draft: draft('pending'),
      stack: addAnnotatePin(session.stack, draft('saved'))
    }
    session = endAnnotateMode(session)

    expect(session.mode).toBe(false)
    expect(session.overlayLive).toBe(false)
    expect(session.draft).toBeNull()
    expect(session.stack.pins).toHaveLength(1)
  })
})

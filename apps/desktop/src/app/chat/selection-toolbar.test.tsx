import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { type RefObject, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $sideChatRequest } from '@/store/side-chat'

import { SelectionToolbar } from './selection-toolbar'

afterEach(() => {
  cleanup()
  delete (Range.prototype as Partial<Range>).getBoundingClientRect
})

const MEASURED = {
  bottom: 60,
  height: 20,
  left: 10,
  right: 110,
  toJSON: () => ({}),
  top: 40,
  width: 100,
  x: 10,
  y: 40
} as DOMRect

beforeEach(() => {
  $sideChatRequest.set(null)
  window.getSelection()?.removeAllRanges()
  // jsdom has NO Range.getBoundingClientRect at all (not even a zero one), and
  // the pill requires a real box before it shows — a zero-size range is a
  // collapsed selection. Define it for the duration of the file.
  Object.defineProperty(Range.prototype, 'getBoundingClientRect', { configurable: true, value: () => MEASURED })
})

function Harness({ outside = false }: { outside?: boolean } & Record<string, unknown>) {
  const bounds = useRef<HTMLDivElement | null>(null)

  return (
    <div>
      <div data-stored-session-id="main-1" data-testid="bounds" ref={bounds}>
        <p data-message-id="msg-1">select me</p>
      </div>
      {outside ? <p data-testid="elsewhere">somewhere else</p> : null}
      <SelectionToolbar container={bounds as RefObject<HTMLElement | null>} />
    </div>
  )
}

function select(node: Element) {
  const range = document.createRange()

  range.selectNodeContents(node)
  const selection = window.getSelection()

  selection?.removeAllRanges()
  selection?.addRange(range)
}

describe('SelectionToolbar', () => {
  it('stays out of the way until something is selected', () => {
    render(<Harness />)

    expect(screen.queryByRole('button', { name: /chat about selection/i })).toBeNull()
  })

  it('offers the side-chat verb for a selection, naming the message it came from', () => {
    const { getByText } = render(<Harness />)

    select(getByText('select me'))
    fireEvent.mouseUp(document)

    fireEvent.click(screen.getByRole('button', { name: /chat about selection/i }))

    expect($sideChatRequest.get()).toEqual({
      fromStoredSessionId: 'main-1',
      messageId: 'msg-1',
      text: 'select me'
    })
  })

  it('carries the transcript’s own conversation, not whatever session is selected', () => {
    // The transcript declares which conversation it renders; the request must
    // name that one so the side chat is attributed and routed to it.
    const { getByText } = render(<Harness />)

    select(getByText('select me'))
    fireEvent.mouseUp(document)
    fireEvent.click(screen.getByRole('button', { name: /chat about selection/i }))

    expect($sideChatRequest.get()?.fromStoredSessionId).toBe('main-1')
  })

  it('ignores a selection that belongs to another surface, so tiles keep their gestures', () => {
    const { getByTestId } = render(<Harness outside />)

    select(getByTestId('elsewhere'))
    fireEvent.mouseUp(document)

    expect(screen.queryByRole('button', { name: /chat about selection/i })).toBeNull()
  })

  it('gets out of the way on Escape', () => {
    const { getByText } = render(<Harness />)

    select(getByText('select me'))
    fireEvent.mouseUp(document)
    expect(screen.getByRole('button', { name: /chat about selection/i })).toBeTruthy()

    fireEvent.keyDown(document, { key: 'Escape' })

    expect(screen.queryByRole('button', { name: /chat about selection/i })).toBeNull()
  })

  it('survives the click that lands on it rather than dismissing itself first', () => {
    const { getByText } = render(<Harness />)

    select(getByText('select me'))
    fireEvent.mouseUp(document)

    // The pill's own mousedown precedes its click; dismissing there would unmount
    // the button mid-gesture and the request would never be posted.
    fireEvent.mouseDown(screen.getByRole('button', { name: /chat about selection/i }))
    fireEvent.click(screen.getByRole('button', { name: /chat about selection/i }))

    expect($sideChatRequest.get()?.text).toBe('select me')
  })
})

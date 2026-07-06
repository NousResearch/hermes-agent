import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import type * as React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  HERMES_SESSION_MIME,
  HERMES_SESSION_PINNED_MIME,
  readSessionDrag,
  type SessionDragPayload,
  writeSessionDrag
} from '@/app/chat/composer/inline-refs'

import {
  placeSessionIdAtAnchor,
  previewItemsAtAnchor,
  sessionDropAnchor,
  useSessionDropZone
} from './use-session-drop-zone'

// jsdom has no DataTransfer; a plain object with the same surface is enough
// for both the writer (setData/effectAllowed) and the zone (types/getData).
function fakeTransfer(data: Record<string, string> = {}) {
  const store = { ...data }

  return {
    dropEffect: 'none',
    effectAllowed: 'uninitialized',
    getData: (type: string) => store[type] ?? '',
    setData: (type: string, value: string) => {
      store[type] = value
    },
    get types() {
      return Object.keys(store)
    }
  } as unknown as DataTransfer
}

function sessionTransfer(payload: SessionDragPayload) {
  const transfer = fakeTransfer()
  writeSessionDrag(transfer, payload)

  return transfer
}

const UNPINNED_ROW: SessionDragPayload = {
  id: 'live-1',
  pinId: 'root-1',
  pinned: false,
  profile: 'default',
  title: 'Recent session'
}

const PINNED_ROW: SessionDragPayload = {
  id: 'live-2',
  pinId: 'root-2',
  pinned: true,
  profile: 'default',
  title: 'Pinned session'
}

function Probe({
  acceptPinned,
  children,
  draggingSessionId,
  onDropSession
}: {
  acceptPinned: boolean
  children?: React.ReactNode
  draggingSessionId?: null | string
  onDropSession: (session: SessionDragPayload) => void
}) {
  const { active, anchor, dropHandlers } = useSessionDropZone({
    accepts: flags => flags.pinned === acceptPinned,
    draggingSessionId,
    onDropSession
  })

  return (
    <div
      data-active={active ? 'true' : 'false'}
      data-anchor={anchor ? `${anchor.before ? 'before' : 'after'}:${anchor.sessionId}` : ''}
      data-testid="zone"
      {...dropHandlers}
    >
      <span data-testid="zone-child">rows</span>
      {children}
    </div>
  )
}

function dragEvent(type: string, dataTransfer: DataTransfer, clientY = 0) {
  const event = new Event(type, {
    bubbles: true,
    cancelable: true
  })

  Object.defineProperty(event, 'clientY', { value: clientY })
  Object.defineProperty(event, 'dataTransfer', { value: dataTransfer })

  return event
}

afterEach(cleanup)

describe('writeSessionDrag / readSessionDrag pin metadata', () => {
  it('round-trips pinId and pinned, flagging pinned drags as a readable type', () => {
    const transfer = sessionTransfer(PINNED_ROW)

    expect(Array.from(transfer.types)).toContain(HERMES_SESSION_MIME)
    expect(Array.from(transfer.types)).toContain(HERMES_SESSION_PINNED_MIME)
    expect(readSessionDrag(transfer)).toEqual(PINNED_ROW)
  })

  it('omits the pinned marker for unpinned rows', () => {
    const transfer = sessionTransfer(UNPINNED_ROW)

    expect(Array.from(transfer.types)).not.toContain(HERMES_SESSION_PINNED_MIME)
    expect(readSessionDrag(transfer)).toEqual(UNPINNED_ROW)
  })

  it('falls back to the live id as pinId for payloads written without one', () => {
    const transfer = fakeTransfer({
      [HERMES_SESSION_MIME]: JSON.stringify({ id: 'legacy-1', profile: 'default', title: 'Old payload' })
    })

    expect(readSessionDrag(transfer)).toEqual({
      id: 'legacy-1',
      pinId: 'legacy-1',
      pinned: false,
      profile: 'default',
      title: 'Old payload'
    })
  })
})

describe('useSessionDropZone', () => {
  it('accepts an unpinned row drag when acceptPinned=false and drops it', () => {
    const onDropSession = vi.fn()
    render(<Probe acceptPinned={false} onDropSession={onDropSession} />)
    const zone = screen.getByTestId('zone')
    const transfer = sessionTransfer(UNPINNED_ROW)

    fireEvent.dragEnter(zone, { dataTransfer: transfer })
    expect(zone.dataset.active).toBe('true')

    // preventDefault on dragover is what makes the browser allow the drop.
    expect(fireEvent.dragOver(zone, { dataTransfer: transfer })).toBe(false)

    fireEvent.drop(zone, { dataTransfer: transfer })
    // The drop event rides along so handlers can compute the drop position.
    expect(onDropSession).toHaveBeenCalledWith(UNPINNED_ROW, expect.objectContaining({ type: 'drop' }), null)
    expect(zone.dataset.active).toBe('false')
  })

  it('accepts a pinned row drag when acceptPinned=true and drops it', () => {
    const onDropSession = vi.fn()
    render(<Probe acceptPinned onDropSession={onDropSession} />)
    const zone = screen.getByTestId('zone')
    const transfer = sessionTransfer(PINNED_ROW)

    fireEvent.dragEnter(zone, { dataTransfer: transfer })
    expect(zone.dataset.active).toBe('true')

    fireEvent.drop(zone, { dataTransfer: transfer })
    expect(onDropSession).toHaveBeenCalledWith(PINNED_ROW, expect.objectContaining({ type: 'drop' }), null)
  })

  it('ignores drags whose pin-state it would not act on', () => {
    const onDropSession = vi.fn()
    render(<Probe acceptPinned={false} onDropSession={onDropSession} />)
    const zone = screen.getByTestId('zone')
    const transfer = sessionTransfer(PINNED_ROW)

    fireEvent.dragEnter(zone, { dataTransfer: transfer })
    expect(zone.dataset.active).toBe('false')

    // No preventDefault → the drop stays disallowed for this zone.
    expect(fireEvent.dragOver(zone, { dataTransfer: transfer })).toBe(true)

    fireEvent.drop(zone, { dataTransfer: transfer })
    expect(onDropSession).not.toHaveBeenCalled()
  })

  it('ignores non-session drags entirely', () => {
    const onDropSession = vi.fn()
    render(<Probe acceptPinned={false} onDropSession={onDropSession} />)
    const zone = screen.getByTestId('zone')
    const transfer = fakeTransfer({ 'text/plain': 'not a session' })

    fireEvent.dragEnter(zone, { dataTransfer: transfer })
    expect(zone.dataset.active).toBe('false')

    fireEvent.drop(zone, { dataTransfer: transfer })
    expect(onDropSession).not.toHaveBeenCalled()
  })

  it('keeps the highlight while moving across nested children', () => {
    render(<Probe acceptPinned={false} onDropSession={vi.fn()} />)
    const zone = screen.getByTestId('zone')
    const child = screen.getByTestId('zone-child')
    const transfer = sessionTransfer(UNPINNED_ROW)

    fireEvent.dragEnter(zone, { dataTransfer: transfer })
    fireEvent.dragEnter(child, { dataTransfer: transfer })
    fireEvent.dragLeave(zone, { dataTransfer: transfer })
    expect(zone.dataset.active).toBe('true')

    fireEvent.dragLeave(child, { dataTransfer: transfer })
    expect(zone.dataset.active).toBe('false')
  })

  it('does not wedge after stray leave events from unaccepted drags', () => {
    render(<Probe acceptPinned={false} onDropSession={vi.fn()} />)
    const zone = screen.getByTestId('zone')
    const pinned = sessionTransfer(PINNED_ROW)

    // A drag this zone ignores still emits leave events on the way out.
    fireEvent.dragEnter(zone, { dataTransfer: pinned })
    fireEvent.dragLeave(zone, { dataTransfer: pinned })
    fireEvent.dragLeave(zone, { dataTransfer: pinned })

    const accepted = sessionTransfer(UNPINNED_ROW)
    fireEvent.dragEnter(zone, { dataTransfer: accepted })
    expect(zone.dataset.active).toBe('true')

    fireEvent.dragLeave(zone, { dataTransfer: accepted })
    expect(zone.dataset.active).toBe('false')
  })
})

describe('session drop ordering helpers', () => {
  it('places a row after the anchor without drifting when moving downward', () => {
    expect(placeSessionIdAtAnchor(['a', 'b', 'c'], 'a', { before: false, sessionId: 'b' })).toEqual(['b', 'a', 'c'])
  })

  it('places a row before the anchor when moving upward', () => {
    expect(placeSessionIdAtAnchor(['a', 'b', 'c'], 'c', { before: true, sessionId: 'b' })).toEqual(['a', 'c', 'b'])
  })

  it('returns null for no-op anchors', () => {
    expect(placeSessionIdAtAnchor(['a', 'b'], 'a', null)).toBeNull()
    expect(placeSessionIdAtAnchor(['a', 'b'], 'a', { before: true, sessionId: 'a' })).toBeNull()
    expect(placeSessionIdAtAnchor(['a', 'b'], 'a', { before: true, sessionId: 'missing' })).toBeNull()
  })

  it('previews a cross-section row at the hovered slot', () => {
    expect(
      previewItemsAtAnchor([{ id: 'a' }, { id: 'b' }], { id: 'x' }, { before: false, sessionId: 'a' }).map(
        item => item.id
      )
    ).toEqual(['a', 'x', 'b'])
  })
})

describe('sessionDropAnchor', () => {
  // jsdom layout is all zeros; give the row a real box so the midpoint
  // math has something to bisect.
  function rowWithRect(sessionId: string, top: number, height: number) {
    const row = document.createElement('div')
    row.dataset.sessionId = sessionId
    row.getBoundingClientRect = () =>
      ({ bottom: top + height, height, left: 0, right: 240, top, width: 240 }) as DOMRect
    document.body.appendChild(row)

    return row
  }

  function dropEventAt(target: Element, clientY: number) {
    return { clientY, target } as unknown as React.DragEvent
  }

  function scopedDropEvent(container: Element, target: Element, clientY: number) {
    return { clientY, currentTarget: container, target } as unknown as React.DragEvent
  }

  afterEach(() => {
    document.body.innerHTML = ''
  })

  it('anchors before the row when the pointer is in its top half', () => {
    const row = rowWithRect('s-anchor', 100, 26)

    expect(sessionDropAnchor(dropEventAt(row, 105))).toEqual({ before: true, sessionId: 's-anchor' })
  })

  it('anchors after the row when the pointer is in its bottom half', () => {
    const row = rowWithRect('s-anchor', 100, 26)

    expect(sessionDropAnchor(dropEventAt(row, 120))).toEqual({ before: false, sessionId: 's-anchor' })
  })

  it('resolves through nested children to the enclosing row', () => {
    const row = rowWithRect('s-nested', 50, 26)
    const child = document.createElement('span')
    row.appendChild(child)

    expect(sessionDropAnchor(dropEventAt(child, 52))).toEqual({ before: true, sessionId: 's-nested' })
  })

  it('returns null for drops outside any session row (header / empty space)', () => {
    const header = document.createElement('div')
    document.body.appendChild(header)

    expect(sessionDropAnchor(dropEventAt(header, 10))).toBeNull()
  })

  it('keeps the previous anchor while the pointer is in a row middle band', () => {
    const row = rowWithRect('s-anchor', 100, 28)
    const previous = { before: true, sessionId: 's-prev' }

    expect(sessionDropAnchor(dropEventAt(row, 114), { previous })).toEqual(previous)
    expect(sessionDropAnchor(dropEventAt(row, 101), { previous })).toEqual({ before: true, sessionId: 's-anchor' })
    expect(sessionDropAnchor(dropEventAt(row, 127), { previous })).toEqual({ before: false, sessionId: 's-anchor' })
  })

  it('ignores the dragged row as a hover target to avoid animated snapback', () => {
    const container = document.createElement('div')
    const moving = rowWithRect('moving', 100, 28)
    const target = rowWithRect('target', 132, 28)
    container.append(moving, target)
    document.body.appendChild(container)

    expect(sessionDropAnchor(scopedDropEvent(container, moving, 114), { movingSessionId: 'moving' })).toEqual({
      before: true,
      sessionId: 'target'
    })
  })
})

describe('useSessionDropZone stable anchor state', () => {
  it('drops with the last stable anchor when the dragged row is under the pointer', () => {
    const onDropSession = vi.fn()
    const transfer = sessionTransfer(UNPINNED_ROW)
    const targetRect = { bottom: 128, height: 28, left: 0, right: 240, top: 100, width: 240 } as DOMRect
    const movingRect = { bottom: 160, height: 28, left: 0, right: 240, top: 132, width: 240 } as DOMRect

    render(
      <Probe acceptPinned={false} draggingSessionId="live-1" onDropSession={onDropSession}>
        <div
          data-session-id="target"
          ref={node => {
            if (node) {
              node.getBoundingClientRect = () => targetRect
            }
          }}
        />
        <div
          data-session-id="live-1"
          ref={node => {
            if (node) {
              node.getBoundingClientRect = () => movingRect
            }
          }}
        />
      </Probe>
    )

    const zone = screen.getByTestId('zone')
    fireEvent(zone, dragEvent('dragenter', transfer, 127))
    expect(zone.dataset.anchor).toBe('after:target')

    fireEvent(zone, dragEvent('dragover', transfer, 140))
    expect(zone.dataset.anchor).toBe('after:target')

    fireEvent(zone, dragEvent('drop', transfer, 140))
    expect(onDropSession).toHaveBeenCalledWith(UNPINNED_ROW, expect.objectContaining({ type: 'drop' }), {
      before: false,
      sessionId: 'target'
    })
  })

  it('clears section highlight when local session drag ends without a matching leave', () => {
    const transfer = sessionTransfer(UNPINNED_ROW)

    const { rerender } = render(<Probe acceptPinned={false} draggingSessionId="live-1" onDropSession={vi.fn()} />)

    const zone = screen.getByTestId('zone')

    fireEvent.dragEnter(zone, { dataTransfer: transfer })
    expect(zone.dataset.active).toBe('true')

    rerender(<Probe acceptPinned={false} draggingSessionId={null} onDropSession={vi.fn()} />)
    expect(zone.dataset.active).toBe('false')
  })
})

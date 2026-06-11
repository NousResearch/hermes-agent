import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import type * as React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  HERMES_SESSION_ARCHIVED_MIME,
  HERMES_SESSION_MIME,
  HERMES_SESSION_PINNED_MIME,
  readSessionDrag,
  type SessionDragPayload,
  writeSessionDrag
} from '@/app/chat/composer/inline-refs'

import {
  placeSessionIdAtAnchor,
  previewItemsAtAnchor,
  type SessionDragFlags,
  type SessionDropAnchor,
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

function dragEvent(type: string, dataTransfer: DataTransfer, clientY: number) {
  const event = new Event(type, { bubbles: true, cancelable: true }) as DragEvent
  Object.defineProperty(event, 'clientY', { value: clientY })
  Object.defineProperty(event, 'dataTransfer', { value: dataTransfer })

  return event
}

const UNPINNED_ROW: SessionDragPayload = {
  archived: false,
  id: 'live-1',
  pinId: 'root-1',
  pinned: false,
  profile: 'default',
  title: 'Recent session'
}

const PINNED_ROW: SessionDragPayload = {
  archived: false,
  id: 'live-2',
  pinId: 'root-2',
  pinned: true,
  profile: 'default',
  title: 'Pinned session'
}

const ARCHIVED_ROW: SessionDragPayload = {
  archived: true,
  id: 'live-3',
  pinId: 'root-3',
  pinned: false,
  profile: 'default',
  title: 'Archived session'
}

// Representative zone predicates for the reusable hook.
const PINNED_ZONE = (flags: SessionDragFlags) => !flags.pinned && !flags.archived
const SESSIONS_ZONE = (flags: SessionDragFlags) => flags.pinned || flags.archived
const ARCHIVED_ZONE = (flags: SessionDragFlags) => !flags.archived

function Probe({
  accepts,
  children,
  draggingSessionId,
  onDropSession
}: {
  accepts: (flags: SessionDragFlags) => boolean
  children?: React.ReactNode
  draggingSessionId?: string
  onDropSession: (session: SessionDragPayload, event: React.DragEvent, anchor: null | SessionDropAnchor) => void
}) {
  const { active, dropHandlers } = useSessionDropZone({ accepts, draggingSessionId, onDropSession })

  return (
    <div data-active={active ? 'true' : 'false'} data-testid="zone" {...dropHandlers}>
      {children ?? <span data-testid="zone-child">rows</span>}
    </div>
  )
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
      archived: false,
      id: 'legacy-1',
      pinId: 'legacy-1',
      pinned: false,
      profile: 'default',
      title: 'Old payload'
    })
  })

  it('flags Archived-section drags as a readable type', () => {
    const transfer = sessionTransfer(ARCHIVED_ROW)

    expect(Array.from(transfer.types)).toContain(HERMES_SESSION_ARCHIVED_MIME)
    expect(readSessionDrag(transfer)).toEqual(ARCHIVED_ROW)
  })
})

describe('useSessionDropZone', () => {
  it('accepts an unpinned row drag in the Pinned zone and drops it', () => {
    const onDropSession = vi.fn()
    render(<Probe accepts={PINNED_ZONE} onDropSession={onDropSession} />)
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

  it('accepts a pinned row drag in the Sessions zone and drops it', () => {
    const onDropSession = vi.fn()
    render(<Probe accepts={SESSIONS_ZONE} onDropSession={onDropSession} />)
    const zone = screen.getByTestId('zone')
    const transfer = sessionTransfer(PINNED_ROW)

    fireEvent.dragEnter(zone, { dataTransfer: transfer })
    expect(zone.dataset.active).toBe('true')

    fireEvent.drop(zone, { dataTransfer: transfer })
    expect(onDropSession).toHaveBeenCalledWith(PINNED_ROW, expect.objectContaining({ type: 'drop' }), null)
  })

  it('restores an archived row dropped on the Sessions zone, and keeps it out of Pinned', () => {
    const onDropSession = vi.fn()
    const { unmount } = render(<Probe accepts={SESSIONS_ZONE} onDropSession={onDropSession} />)
    const sessionsZone = screen.getByTestId('zone')
    const transfer = sessionTransfer(ARCHIVED_ROW)

    fireEvent.dragEnter(sessionsZone, { dataTransfer: transfer })
    expect(sessionsZone.dataset.active).toBe('true')

    fireEvent.drop(sessionsZone, { dataTransfer: transfer })
    expect(onDropSession).toHaveBeenCalledWith(ARCHIVED_ROW, expect.objectContaining({ type: 'drop' }), null)
    unmount()

    // The Pinned zone must never light up for an archived drag — a pin can't
    // resolve an archived row.
    const onPinnedDrop = vi.fn()
    render(<Probe accepts={PINNED_ZONE} onDropSession={onPinnedDrop} />)
    const pinnedZone = screen.getByTestId('zone')

    fireEvent.dragEnter(pinnedZone, { dataTransfer: sessionTransfer(ARCHIVED_ROW) })
    expect(pinnedZone.dataset.active).toBe('false')
  })

  it('archives a live row dropped on the Archived zone but ignores archived rows there', () => {
    const onDropSession = vi.fn()
    render(<Probe accepts={ARCHIVED_ZONE} onDropSession={onDropSession} />)
    const zone = screen.getByTestId('zone')

    fireEvent.dragEnter(zone, { dataTransfer: sessionTransfer(PINNED_ROW) })
    expect(zone.dataset.active).toBe('true')
    fireEvent.drop(zone, { dataTransfer: sessionTransfer(PINNED_ROW) })
    expect(onDropSession).toHaveBeenCalledWith(PINNED_ROW, expect.objectContaining({ type: 'drop' }), null)

    fireEvent.dragEnter(zone, { dataTransfer: sessionTransfer(ARCHIVED_ROW) })
    expect(zone.dataset.active).toBe('false')
  })

  it('ignores drags whose pin-state it would not act on', () => {
    const onDropSession = vi.fn()
    render(<Probe accepts={PINNED_ZONE} onDropSession={onDropSession} />)
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
    render(<Probe accepts={PINNED_ZONE} onDropSession={onDropSession} />)
    const zone = screen.getByTestId('zone')
    const transfer = fakeTransfer({ 'text/plain': 'not a session' })

    fireEvent.dragEnter(zone, { dataTransfer: transfer })
    expect(zone.dataset.active).toBe('false')

    fireEvent.drop(zone, { dataTransfer: transfer })
    expect(onDropSession).not.toHaveBeenCalled()
  })

  it('keeps the highlight while moving across nested children', () => {
    render(<Probe accepts={PINNED_ZONE} onDropSession={vi.fn()} />)
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
    render(<Probe accepts={PINNED_ZONE} onDropSession={vi.fn()} />)
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

  it('drops with the last stable anchor when the animated dragged row is under the pointer', () => {
    const onDropSession = vi.fn()
    const movingRow = { ...UNPINNED_ROW, id: 'moving' }
    const transfer = sessionTransfer(movingRow)

    render(
      <Probe accepts={PINNED_ZONE} draggingSessionId="moving" onDropSession={onDropSession}>
        <div data-session-id="a" data-testid="row-a" />
        <div data-session-id="moving" data-testid="row-moving" />
        <div data-session-id="b" data-testid="row-b" />
      </Probe>
    )

    const rowA = screen.getByTestId('row-a')
    const moving = screen.getByTestId('row-moving')
    const rowB = screen.getByTestId('row-b')

    rowA.getBoundingClientRect = () => ({ bottom: 126, height: 26, left: 0, right: 240, top: 100, width: 240 }) as DOMRect
    moving.getBoundingClientRect = () =>
      ({ bottom: 153, height: 26, left: 0, right: 240, top: 127, width: 240 }) as DOMRect
    rowB.getBoundingClientRect = () => ({ bottom: 180, height: 26, left: 0, right: 240, top: 154, width: 240 }) as DOMRect

    fireEvent(rowA, dragEvent('dragenter', transfer, 124))
    fireEvent(moving, dragEvent('dragover', transfer, 140))
    fireEvent(moving, dragEvent('drop', transfer, 140))

    expect(onDropSession).toHaveBeenCalledWith(movingRow, expect.objectContaining({ type: 'drop' }), {
      before: false,
      sessionId: 'a'
    })
  })

  it('clears the section highlight when a local session drag ends without a matching leave', () => {
    const onDropSession = vi.fn()

    const { rerender } = render(
      <Probe accepts={PINNED_ZONE} draggingSessionId="live-1" onDropSession={onDropSession} />
    )

    const zone = screen.getByTestId('zone')

    fireEvent.dragEnter(zone, { dataTransfer: sessionTransfer(UNPINNED_ROW) })

    expect(zone.dataset.active).toBe('true')

    rerender(<Probe accepts={PINNED_ZONE} onDropSession={onDropSession} />)
    expect(zone.dataset.active).toBe('false')
  })
})

describe('sessionDropAnchor', () => {
  // jsdom layout is all zeros; give the row a real box so the midpoint
  // math has something to bisect.
  function rowWithRect(sessionId: string, top: number, height: number, parent: HTMLElement = document.body) {
    const row = document.createElement('div')
    row.dataset.sessionId = sessionId
    row.getBoundingClientRect = () =>
      ({ bottom: top + height, height, left: 0, right: 240, top, width: 240 }) as DOMRect
    parent.appendChild(row)

    return row
  }

  function dropEventAt(target: Element, clientY: number, currentTarget?: Element) {
    return { clientY, currentTarget, target } as unknown as React.DragEvent
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

  it('keeps the previous anchor through a row middle band and switches only near the edge', () => {
    const root = document.createElement('div')
    const previous = { before: false, sessionId: 'a' }
    document.body.appendChild(root)
    rowWithRect('a', 100, 26, root)
    const rowB = rowWithRect('b', 127, 26, root)

    expect(sessionDropAnchor(dropEventAt(rowB, 140, root), { previous })).toEqual(previous)
    expect(sessionDropAnchor(dropEventAt(rowB, 150, root), { previous })).toEqual({ before: false, sessionId: 'b' })
  })

  it('ignores the dragged row as a hover target so animated previews do not snap back', () => {
    const root = document.createElement('div')
    const previous = { before: false, sessionId: 'a' }
    document.body.appendChild(root)
    rowWithRect('a', 100, 26, root)
    const moving = rowWithRect('moving', 127, 26, root)
    rowWithRect('b', 154, 26, root)

    expect(sessionDropAnchor(dropEventAt(moving, 140, root), { movingSessionId: 'moving', previous })).toEqual(previous)
  })
})

describe('placeSessionIdAtAnchor', () => {
  it('moves an existing id before the anchor', () => {
    expect(placeSessionIdAtAnchor(['a', 'b', 'c', 'd'], 'c', { before: true, sessionId: 'a' })).toEqual([
      'c',
      'a',
      'b',
      'd'
    ])
  })

  it('moves an existing id after the anchor', () => {
    expect(placeSessionIdAtAnchor(['a', 'b', 'c', 'd'], 'a', { before: false, sessionId: 'c' })).toEqual([
      'b',
      'c',
      'a',
      'd'
    ])
  })

  it('inserts a cross-section id at the exact anchor position', () => {
    expect(placeSessionIdAtAnchor(['a', 'b', 'c'], 'pinned-row', { before: true, sessionId: 'b' })).toEqual([
      'a',
      'pinned-row',
      'b',
      'c'
    ])
  })

  it('returns null for self-drops, missing anchors, and missing anchor ids', () => {
    expect(placeSessionIdAtAnchor(['a', 'b'], 'a', { before: true, sessionId: 'a' })).toBeNull()
    expect(placeSessionIdAtAnchor(['a', 'b'], 'c', null)).toBeNull()
    expect(placeSessionIdAtAnchor(['a', 'b'], 'c', { before: false, sessionId: 'missing' })).toBeNull()
  })
})

describe('previewItemsAtAnchor', () => {
  const items = [{ id: 'a' }, { id: 'b' }, { id: 'c' }]

  it('previews same-section reorder by moving the dragged item before the anchor', () => {
    expect(previewItemsAtAnchor(items, items[2], { before: true, sessionId: 'a' })).toEqual([
      { id: 'c' },
      { id: 'a' },
      { id: 'b' }
    ])
  })

  it('previews a cross-section insert without mutating the original list', () => {
    const moving = { id: 'pinned-row' }

    expect(previewItemsAtAnchor(items, moving, { before: false, sessionId: 'b' })).toEqual([
      { id: 'a' },
      { id: 'b' },
      moving,
      { id: 'c' }
    ])
    expect(items).toEqual([{ id: 'a' }, { id: 'b' }, { id: 'c' }])
  })

  it('leaves the list unchanged when no usable preview target exists', () => {
    expect(previewItemsAtAnchor(items, null, { before: true, sessionId: 'a' })).toBe(items)
    expect(previewItemsAtAnchor(items, items[0], null)).toBe(items)
    expect(previewItemsAtAnchor(items, items[0], { before: true, sessionId: 'missing' })).toBe(items)
    expect(previewItemsAtAnchor(items, items[0], { before: false, sessionId: 'a' })).toBe(items)
  })
})

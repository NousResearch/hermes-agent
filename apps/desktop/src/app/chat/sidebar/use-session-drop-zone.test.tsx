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
  type FrozenSectionBand,
  frozenSectionKeyFromPoint,
  placeSessionIdAtAnchor,
  previewItemsAtAnchor,
  previewItemsForSessionDrop,
  type SessionDragFlags,
  type SessionDropAnchor,
  sessionDropAnchor,
  sessionDropMarkerIndex,
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
  draggingSession,
  draggingSessionId,
  onDropSession
}: {
  accepts: (flags: SessionDragFlags) => boolean
  children?: React.ReactNode
  draggingSession?: null | SessionDragPayload
  draggingSessionId?: string
  onDropSession: (session: SessionDragPayload, event: React.DragEvent, anchor: null | SessionDropAnchor) => void
}) {
  const { active, dropHandlers } = useSessionDropZone({ accepts, draggingSession, draggingSessionId, onDropSession })

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

  it('accepts the active local session drag when DataTransfer hides custom MIME types during hover', () => {
    const onDropSession = vi.fn()
    render(<Probe accepts={PINNED_ZONE} draggingSession={UNPINNED_ROW} onDropSession={onDropSession} />)
    const zone = screen.getByTestId('zone')
    const opaqueTransfer = fakeTransfer()

    fireEvent.dragEnter(zone, { dataTransfer: opaqueTransfer })
    expect(zone.dataset.active).toBe('true')

    expect(fireEvent.dragOver(zone, { dataTransfer: opaqueTransfer })).toBe(false)

    fireEvent.drop(zone, { dataTransfer: opaqueTransfer })
    expect(onDropSession).toHaveBeenCalledWith(UNPINNED_ROW, expect.objectContaining({ type: 'drop' }), null)
    expect(zone.dataset.active).toBe('false')
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

  it('drops by the physical gap midpoint when the animated dragged row is under the pointer', () => {
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
      before: true,
      sessionId: 'b'
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

  it('switches at the physical row midpoint even when there was a previous anchor', () => {
    const root = document.createElement('div')
    const previous = { before: false, sessionId: 'a' }
    document.body.appendChild(root)
    rowWithRect('a', 100, 26, root)
    const rowB = rowWithRect('b', 127, 26, root)

    expect(sessionDropAnchor(dropEventAt(rowB, 139, root), { previous })).toEqual({ before: true, sessionId: 'b' })
    expect(sessionDropAnchor(dropEventAt(rowB, 150, root), { previous })).toEqual({ before: false, sessionId: 'b' })
  })

  it('lets an adjacent move switch sides after the same target row midpoint', () => {
    const root = document.createElement('div')
    const previous = { before: true, sessionId: 'b' }
    document.body.appendChild(root)
    const rowB = rowWithRect('b', 127, 26, root)

    expect(sessionDropAnchor(dropEventAt(rowB, 141, root), { previous })).toEqual({ before: false, sessionId: 'b' })
  })

  it('ignores the dragged row as a hover target and resolves its gap by pointer position', () => {
    const root = document.createElement('div')
    const previous = { before: false, sessionId: 'a' }
    document.body.appendChild(root)
    rowWithRect('a', 100, 26, root)
    const moving = rowWithRect('moving', 127, 26, root)
    rowWithRect('b', 154, 26, root)

    expect(sessionDropAnchor(dropEventAt(moving, 140, root), { movingSessionId: 'moving', previous })).toEqual({
      before: true,
      sessionId: 'b'
    })
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

describe('previewItemsForSessionDrop', () => {
  const items = [{ id: 'a' }, { id: 'b' }, { id: 'c' }]
  const moving = { id: 'moving' }
  const anchor = { before: true, sessionId: 'b' }

  it('previews pointer drags by placing the active row in its destination slot', () => {
    expect(previewItemsForSessionDrop(items, moving, anchor, { active: true, mode: 'pointer' })).toEqual([
      { id: 'a' },
      moving,
      { id: 'b' },
      { id: 'c' }
    ])
  })

  it('previews section-level pointer drops at the end when the item is entering the section', () => {
    expect(previewItemsForSessionDrop(items, moving, null, { active: true, mode: 'pointer' })).toEqual([
      { id: 'a' },
      { id: 'b' },
      { id: 'c' },
      moving
    ])
  })

  it('still previews native HTML drags where no sortable overlay owns the row motion', () => {
    expect(previewItemsForSessionDrop(items, moving, anchor, { active: true, mode: 'native' })).toEqual([
      { id: 'a' },
      moving,
      { id: 'b' },
      { id: 'c' }
    ])
  })

  it('leaves inactive zones untouched', () => {
    expect(previewItemsForSessionDrop(items, moving, anchor, { active: false, mode: 'native' })).toBe(items)
  })
})

describe('sessionDropMarkerIndex', () => {
  it('places the marker before or after the anchored row', () => {
    expect(sessionDropMarkerIndex(['a', 'b', 'c'], { before: true, sessionId: 'b' })).toBe(1)
    expect(sessionDropMarkerIndex(['a', 'b', 'c'], { before: false, sessionId: 'b' })).toBe(2)
  })

  it('falls back to the end for section-level drops without a row anchor', () => {
    expect(sessionDropMarkerIndex(['a', 'b'], null)).toBe(2)
    expect(sessionDropMarkerIndex(['a', 'b'], { before: true, sessionId: 'missing' })).toBe(2)
    expect(sessionDropMarkerIndex([], null)).toBe(0)
  })
})

describe('frozenSectionKeyFromPoint', () => {
  // Pinned 100–200, a 10px gap, Sessions 210–300 (mirrors the stacked sidebar
  // sections; the gap is the small padding between section roots).
  const bands: FrozenSectionBand[] = [
    { bottom: 200, key: 'pinned', top: 100 },
    { bottom: 300, key: 'sessions', top: 210 }
  ]

  it('returns the band that contains the pointer', () => {
    expect(frozenSectionKeyFromPoint(bands, 150)).toBe('pinned')
    expect(frozenSectionKeyFromPoint(bands, 250)).toBe('sessions')
    expect(frozenSectionKeyFromPoint(bands, 100)).toBe('pinned')
    expect(frozenSectionKeyFromPoint(bands, 300)).toBe('sessions')
  })

  it('snaps a pointer in the gap between bands to the nearer band', () => {
    expect(frozenSectionKeyFromPoint(bands, 203)).toBe('pinned')
    expect(frozenSectionKeyFromPoint(bands, 207)).toBe('sessions')
  })

  it('returns null outside the whole sections span', () => {
    expect(frozenSectionKeyFromPoint(bands, 50)).toBeNull()
    expect(frozenSectionKeyFromPoint(bands, 400)).toBeNull()
    expect(frozenSectionKeyFromPoint([], 150)).toBeNull()
  })

  it('keeps a stationary pointer on one section regardless of later reflow', () => {
    // The drag-jitter regression: while the pointer holds still inside the
    // frozen Sessions band, the resolved section must not change even though the
    // live layout reflows underneath (which is why bands are frozen, not live).
    const y = 250
    const first = frozenSectionKeyFromPoint(bands, y)
    // A reflow can never move the frozen bands, so re-resolving the same point
    // yields the same section every frame — no oscillation.
    expect(frozenSectionKeyFromPoint(bands, y)).toBe(first)
    expect(first).toBe('sessions')
  })
})

import { type DragEvent as ReactDragEvent, useCallback, useEffect, useRef, useState } from 'react'

import {
  dragHasSession,
  dragSessionIsArchived,
  dragSessionIsPinned,
  readSessionDrag,
  type SessionDragPayload
} from '@/app/chat/composer/inline-refs'

/** Drag-state flags readable during dragover (they ride as marker MIME types;
 * the payload itself is sealed until drop). */
export interface SessionDragFlags {
  pinned: boolean
  archived: boolean
}

interface SessionDropZoneOptions {
  /** Which drags this zone acts on. Pinned/Sessions/Archived each choose their
   * own policy based on the drag's pinned/archived flags. */
  accepts: (flags: SessionDragFlags) => boolean
  /** The native drag payload itself is not reliably readable until drop, so
   * the owner passes the active row id for hover-time self-anchor filtering. */
  draggingSessionId?: null | string
  /** The drop event rides along so handlers can resolve the drop position
   * (see {@link sessionDropAnchor}). */
  onDropSession: (session: SessionDragPayload, event: ReactDragEvent, anchor: null | SessionDropAnchor) => void
}

export interface SessionDropAnchor {
  /** Live session id of the row under the pointer. */
  sessionId: string
  /** True when the dragged row should be inserted before this target row. */
  before: boolean
}

interface SessionDropAnchorOptions {
  /** Row currently being dragged. It should never become its own preview
   * target; when the preview animates under the pointer, keep the old anchor. */
  movingSessionId?: null | string
  /** Previous stable anchor for hysteresis. Used while the pointer is in a
   * row's middle band or over the animated dragged row. */
  previous?: null | SessionDropAnchor
}

const SESSION_DROP_EDGE_BAND_RATIO = 0.38
const SESSION_DROP_MIN_EDGE_BAND_PX = 6
const SESSION_DROP_MAX_EDGE_BAND_PX = 12

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value))
}

export function placeSessionIdAtAnchor(
  ids: readonly string[],
  movingId: string,
  anchor: null | SessionDropAnchor
): null | string[] {
  if (!anchor || anchor.sessionId === movingId) {
    return null
  }

  const next = ids.filter(id => id !== movingId)
  const at = next.indexOf(anchor.sessionId)

  if (at < 0) {
    return null
  }

  next.splice(anchor.before ? at : at + 1, 0, movingId)

  return next
}

export function previewItemsAtAnchor<T extends { id: string }>(
  items: T[],
  movingItem: null | T | undefined,
  anchor: null | SessionDropAnchor
): T[] {
  if (!movingItem || !anchor) {
    return items
  }

  const nextIds = placeSessionIdAtAnchor(
    items.map(item => item.id),
    movingItem.id,
    anchor
  )

  if (!nextIds) {
    return items
  }

  const byId = new Map(items.map(item => [item.id, item]))
  byId.set(movingItem.id, movingItem)

  return nextIds.map(id => byId.get(id)).filter((item): item is T => Boolean(item))
}

function rowAnchorFromRect(
  sessionId: string,
  rect: DOMRect,
  clientY: number,
  previous?: null | SessionDropAnchor
): SessionDropAnchor {
  const edgeBand = clamp(
    rect.height * SESSION_DROP_EDGE_BAND_RATIO,
    SESSION_DROP_MIN_EDGE_BAND_PX,
    SESSION_DROP_MAX_EDGE_BAND_PX
  )

  const topIntent = rect.top + edgeBand
  const bottomIntent = rect.bottom - edgeBand

  if (clientY <= topIntent) {
    return { before: true, sessionId }
  }

  if (clientY >= bottomIntent) {
    return { before: false, sessionId }
  }

  if (previous) {
    return previous
  }

  return { before: clientY < rect.top + rect.height / 2, sessionId }
}

function rowsInScope(event: ReactDragEvent): HTMLElement[] {
  const currentTarget = event.currentTarget as HTMLElement | null

  if (currentTarget?.querySelectorAll) {
    return [...currentTarget.querySelectorAll<HTMLElement>('[data-session-id]')]
  }

  const target = event.target as HTMLElement | null
  const row = target?.closest?.('[data-session-id]') as HTMLElement | null

  return row ? [row] : []
}

/**
 * Resolve a stable insertion anchor from a section-level drag point. The middle
 * of each row is a dead zone that preserves the previous anchor, so animated
 * preview shuffles do not snap back and forth when the pointer pauses.
 */
export function sessionDropAnchor(
  event: ReactDragEvent,
  options: SessionDropAnchorOptions = {}
): null | SessionDropAnchor {
  const rows = rowsInScope(event)
    .map(row => ({ rect: row.getBoundingClientRect(), row, sessionId: row.dataset.sessionId }))
    .filter(
      (entry): entry is { rect: DOMRect; row: HTMLElement; sessionId: string } =>
        Boolean(entry.sessionId) && entry.sessionId !== options.movingSessionId
    )
    .sort((a, b) => a.rect.top - b.rect.top)

  if (rows.length === 0) {
    return options.previous ?? null
  }

  for (const { rect, sessionId } of rows) {
    if (event.clientY >= rect.top && event.clientY <= rect.bottom) {
      return rowAnchorFromRect(sessionId, rect, event.clientY, options.previous)
    }
  }

  for (let index = 0; index < rows.length - 1; index += 1) {
    const current = rows[index]
    const next = rows[index + 1]

    if (event.clientY > current.rect.bottom && event.clientY < next.rect.top) {
      if (options.previous) {
        return options.previous
      }

      const gapMidpoint = current.rect.bottom + (next.rect.top - current.rect.bottom) / 2

      return event.clientY < gapMidpoint
        ? { before: false, sessionId: current.sessionId }
        : { before: true, sessionId: next.sessionId }
    }
  }

  const first = rows[0]
  const last = rows[rows.length - 1]

  if (event.clientY < first.rect.top) {
    return options.previous ?? { before: true, sessionId: first.sessionId }
  }

  if (event.clientY > last.rect.bottom) {
    return options.previous ?? { before: false, sessionId: last.sessionId }
  }

  return options.previous ?? null
}

/**
 * Native drop target for sidebar session rows — the whole row drag already
 * carries `application/x-hermes-session`, so sections can pin, unpin, restore,
 * archive, or reorder without a separate handle.
 *
 * A zone only engages for drags it would act on (Pinned accepts unpinned rows,
 * Sessions accepts pinned rows); other drags never preventDefault, so the
 * cursor honestly reports "no drop here". The enter/leave depth counter keeps
 * nested children from flickering the highlight, mirroring use-file-drop-zone.
 *
 * Spread `dropHandlers` onto the section container; style off `active`.
 */
export function useSessionDropZone({
  accepts: acceptsFlags,
  draggingSessionId,
  onDropSession
}: SessionDropZoneOptions) {
  const [active, setActive] = useState(false)
  const [anchor, setAnchor] = useState<null | SessionDropAnchor>(null)
  const anchorRef = useRef<null | SessionDropAnchor>(null)
  const depth = useRef(0)

  const accepts = useCallback(
    (event: ReactDragEvent) =>
      dragHasSession(event.dataTransfer) &&
      acceptsFlags({
        archived: dragSessionIsArchived(event.dataTransfer),
        pinned: dragSessionIsPinned(event.dataTransfer)
      }),
    [acceptsFlags]
  )

  const reset = useCallback(() => {
    depth.current = 0
    setActive(false)
    setAnchor(null)
    anchorRef.current = null
  }, [])

  useEffect(() => {
    if (!draggingSessionId) {
      reset()
    }
  }, [draggingSessionId, reset])

  const updateAnchor = useCallback(
    (event: ReactDragEvent) => {
      const nextAnchor = sessionDropAnchor(event, {
        movingSessionId: draggingSessionId,
        previous: anchorRef.current
      })

      anchorRef.current = nextAnchor
      setAnchor(nextAnchor)
    },
    [draggingSessionId]
  )

  const onDragEnter = useCallback(
    (event: ReactDragEvent) => {
      if (!accepts(event)) {
        return
      }

      event.preventDefault()
      depth.current += 1
      setActive(true)
      updateAnchor(event)
    },
    [accepts, updateAnchor]
  )

  const onDragOver = useCallback(
    (event: ReactDragEvent) => {
      if (!accepts(event)) {
        return
      }

      event.preventDefault()
      // The row drag advertises effectAllowed='copy' (for composer drops);
      // anything else here would cancel the drop.
      event.dataTransfer.dropEffect = 'copy'
      updateAnchor(event)
    },
    [accepts, updateAnchor]
  )

  // Unaccepted drags never increment, but their leave events still arrive —
  // guard the decrement so they can't drive depth negative and wedge the
  // highlight on a later accepted drag.
  const onDragLeave = useCallback(() => {
    if (depth.current > 0 && --depth.current <= 0) {
      reset()
    }
  }, [reset])

  const onDrop = useCallback(
    (event: ReactDragEvent) => {
      if (!accepts(event)) {
        return
      }

      event.preventDefault()
      const dropAnchor = anchorRef.current
      reset()

      const session = readSessionDrag(event.dataTransfer)

      if (session) {
        onDropSession(session, event, dropAnchor)
      }
    },
    [accepts, onDropSession, reset]
  )

  return {
    anchor,
    active,
    dropHandlers: { onDragEnter, onDragLeave, onDragOver, onDrop }
  }
}

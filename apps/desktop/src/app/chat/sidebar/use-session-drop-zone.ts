import { type DragEvent as ReactDragEvent, useCallback, useRef, useState } from 'react'

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
  /** The drop event rides along so handlers can resolve the drop position
   * (see {@link sessionDropAnchor}). */
  onDropSession: (session: SessionDragPayload, event: ReactDragEvent) => void
}

export interface SessionDropAnchor {
  /** Live session id of the row under the pointer. */
  sessionId: string
  /** True when the pointer sat in the row's top half → insert before it. */
  before: boolean
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

/**
 * Resolve the session row under a drop point (rows carry `data-session-id`)
 * so drop handlers can insert at the pointer position instead of appending.
 * Null for drops on the section header or empty space.
 */
export function sessionDropAnchor(event: ReactDragEvent): null | SessionDropAnchor {
  const target = event.target as HTMLElement | null
  const row = target?.closest?.('[data-session-id]') as HTMLElement | null
  const sessionId = row?.dataset.sessionId

  if (!row || !sessionId) {
    return null
  }

  const rect = row.getBoundingClientRect()

  return { before: event.clientY < rect.top + rect.height / 2, sessionId }
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
export function useSessionDropZone({ accepts: acceptsFlags, onDropSession }: SessionDropZoneOptions) {
  const [active, setActive] = useState(false)
  const [anchor, setAnchor] = useState<null | SessionDropAnchor>(null)
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
  }, [])

  const onDragEnter = useCallback(
    (event: ReactDragEvent) => {
      if (!accepts(event)) {
        return
      }

      event.preventDefault()
      depth.current += 1
      setActive(true)
      setAnchor(sessionDropAnchor(event))
    },
    [accepts]
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
      setAnchor(sessionDropAnchor(event))
    },
    [accepts]
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
      reset()

      const session = readSessionDrag(event.dataTransfer)

      if (session) {
        onDropSession(session, event)
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

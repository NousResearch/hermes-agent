import { type DragEvent as ReactDragEvent, useCallback, useRef, useState } from 'react'

import {
  dragHasSession,
  dragSessionIsPinned,
  readSessionDrag,
  type SessionDragPayload
} from '@/app/chat/composer/inline-refs'

interface SessionDropZoneOptions {
  /** Which drags this zone acts on: pinned rows (true) or unpinned rows (false). */
  acceptPinned: boolean
  onDropSession: (session: SessionDragPayload) => void
}

/**
 * Native drop target for sidebar session rows — the row body's drag already
 * carries `application/x-hermes-session`, so dropping it on the Pinned /
 * Sessions section headers-or-bodies pins and unpins without the context menu.
 *
 * A zone only engages for drags it would act on (Pinned accepts unpinned rows,
 * Sessions accepts pinned rows); other drags never preventDefault, so the
 * cursor honestly reports "no drop here". The enter/leave depth counter keeps
 * nested children from flickering the highlight, mirroring use-file-drop-zone.
 *
 * Spread `dropHandlers` onto the section container; style off `active`.
 */
export function useSessionDropZone({ acceptPinned, onDropSession }: SessionDropZoneOptions) {
  const [active, setActive] = useState(false)
  const depth = useRef(0)

  const accepts = useCallback(
    (event: ReactDragEvent) =>
      dragHasSession(event.dataTransfer) && dragSessionIsPinned(event.dataTransfer) === acceptPinned,
    [acceptPinned]
  )

  const reset = useCallback(() => {
    depth.current = 0
    setActive(false)
  }, [])

  const onDragEnter = useCallback(
    (event: ReactDragEvent) => {
      if (!accepts(event)) {
        return
      }

      event.preventDefault()
      depth.current += 1
      setActive(true)
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
        onDropSession(session)
      }
    },
    [accepts, onDropSession, reset]
  )

  return {
    active,
    dropHandlers: { onDragEnter, onDragLeave, onDragOver, onDrop }
  }
}

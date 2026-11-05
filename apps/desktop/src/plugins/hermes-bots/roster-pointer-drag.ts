import { type DragEvent, type MouseEvent, type PointerEvent, useCallback, useEffect, useRef } from 'react'

import { $draggingBot } from './user-sections'
import { findPointerDropZone, type PointerDropZoneEntry } from './user-sections-ui'

interface RosterPointerDragOptions {
  onOpen: (event: MouseEvent<HTMLButtonElement>) => void
  rosterKey: string
}

interface ActiveRosterPointerDrag {
  cancelled: boolean
  overZone: PointerDropZoneEntry | null
  pointerId: number
  startX: number
  startY: number
}

/**
 * Pointer-driven filing for bot and group rows. macOS 26+ can kill the native
 * HTML5 session immediately after dragstart, so the row owns the whole gesture:
 * threshold, target hit-testing, drop, cancellation and synthetic-click
 * suppression. The existing atom remains the shared visual/drop contract.
 */
export function useRosterPointerDrag({ onOpen, rosterKey }: RosterPointerDragOptions) {
  const pointerDrag = useRef<ActiveRosterPointerDrag | null>(null)
  const suppressClickRef = useRef(false)

  const clearPointerDrag = useCallback(() => {
    pointerDrag.current?.overZone?.setOver(false)
    pointerDrag.current = null

    if ($draggingBot.get() === rosterKey) {
      $draggingBot.set(null)
    }
  }, [rosterKey])

  const cancelPointerDrag = useCallback(
    (pointerId?: number) => {
      if (pointerId !== undefined && pointerDrag.current?.pointerId !== pointerId) {
        return
      }

      clearPointerDrag()
      // Cancellation has no synthetic click to suppress. Leaving this armed
      // would swallow a later Enter/Space-generated click instead.
      suppressClickRef.current = false
    },
    [clearPointerDrag]
  )

  useEffect(() => {
    const cancelOnBlur = () => cancelPointerDrag()

    window.addEventListener('blur', cancelOnBlur)

    return () => {
      window.removeEventListener('blur', cancelOnBlur)
      cancelPointerDrag()
    }
  }, [cancelPointerDrag])

  return {
    onClick: (event: MouseEvent<HTMLButtonElement>) => {
      if (suppressClickRef.current) {
        suppressClickRef.current = false

        return
      }

      onOpen(event)
    },
    // The pointer gesture owns filing; also refuse avatar/image native drags.
    onDragStart: (event: DragEvent<HTMLButtonElement>) => event.preventDefault(),
    onLostPointerCapture: (event: PointerEvent<HTMLButtonElement>) => cancelPointerDrag(event.pointerId),
    onPointerCancel: (event: PointerEvent<HTMLButtonElement>) => cancelPointerDrag(event.pointerId),
    onPointerDown: (event: PointerEvent<HTMLButtonElement>) => {
      if (!event.isPrimary || event.button !== 0) {
        return
      }

      cancelPointerDrag()
      pointerDrag.current = {
        cancelled: false,
        overZone: null,
        pointerId: event.pointerId,
        startX: event.clientX,
        startY: event.clientY
      }

      // Capture keeps move/up targeted at this row after the pointer leaves it.
      try {
        event.currentTarget.setPointerCapture(event.pointerId)
      } catch {
        // Released already; up/cancel/lost-capture still own cleanup.
      }
    },
    onPointerMove: (event: PointerEvent<HTMLButtonElement>) => {
      const st = pointerDrag.current

      if (!st || event.pointerId !== st.pointerId || st.cancelled) {
        return
      }

      if ($draggingBot.get() !== rosterKey) {
        if (suppressClickRef.current) {
          // Escape or another owner ended this gesture; never restart/drop it.
          st.cancelled = true

          return
        }

        if (Math.hypot(event.clientX - st.startX, event.clientY - st.startY) > 6) {
          suppressClickRef.current = true
          $draggingBot.set(rosterKey)
        }

        return
      }

      const zone = findPointerDropZone(event.clientX, event.clientY)

      if (zone !== st.overZone) {
        st.overZone?.setOver(false)
        st.overZone = zone
        zone?.setOver(true)
      }
    },
    onPointerUp: (event: PointerEvent<HTMLButtonElement>) => {
      const st = pointerDrag.current

      if (!st || event.pointerId !== st.pointerId) {
        return
      }

      const inFlight = $draggingBot.get() === rosterKey && !st.cancelled

      if (inFlight) {
        const zone = findPointerDropZone(event.clientX, event.clientY)

        if (zone && !zone.isSource) {
          zone.drop(rosterKey)
          clearPointerDrag()

          return
        }
      }

      // Keep click suppression through pointerup: the browser may still emit
      // one synthetic click even when the drop landed outside a valid zone.
      clearPointerDrag()
    }
  }
}

import { type DragEvent as ReactDragEvent, useCallback, useRef, useState } from 'react'

import { dragHasAttachments } from '@/app/chat/composer/inline-refs'

import { type DroppedFile, extractDroppedFiles, HERMES_PATHS_MIME } from './use-composer-actions'

/** `'session'` is set by callers from the pointer drag session's store —
 *  native drags only ever resolve to `'files'` here (sessions left native
 *  DnD; see session-drag.ts). */
export type DragKind = 'files' | 'session' | null

const dragKindOf = (event: ReactDragEvent): DragKind =>
  dragHasAttachments(event.dataTransfer, HERMES_PATHS_MIME) ? 'files' : null

interface FileDropZoneOptions {
  /** When false the zone ignores drags entirely. */
  enabled?: boolean
  onDropFiles: (files: DroppedFile[]) => void
}

/**
 * "Drop anywhere in this region" affordance for FILE drags — the one drag
 * kind still on native DnD (Finder/OS drops and the project tree must be).
 * An enter/leave depth counter keeps nested children from flickering the
 * active state; `onDropCapture` clears it even when a nested target (the
 * composer) handles the drop and stops propagation before our bubble-phase
 * `onDrop` would fire.
 *
 * Spread `dropHandlers` onto the container; render an overlay off `dragKind`.
 */
export function useFileDropZone({ enabled = true, onDropFiles }: FileDropZoneOptions) {
  const [dragKind, setDragKind] = useState<DragKind>(null)
  const depth = useRef(0)

  const reset = useCallback(() => {
    depth.current = 0
    setDragKind(null)
  }, [])

  const onDragEnter = useCallback(
    (event: ReactDragEvent) => {
      const kind = enabled ? dragKindOf(event) : null

      if (!kind) {
        return
      }

      event.preventDefault()
      depth.current += 1
      setDragKind(kind)
    },
    [enabled]
  )

  const onDragOver = useCallback(
    (event: ReactDragEvent) => {
      if (!enabled || !dragKindOf(event)) {
        return
      }

      event.preventDefault()
      event.dataTransfer.dropEffect = 'copy'
    },
    [enabled]
  )

  const onDragLeave = useCallback(() => {
    if (enabled && --depth.current <= 0) {
      reset()
    }
  }, [enabled, reset])

  const onDrop = useCallback(
    (event: ReactDragEvent) => {
      const kind = enabled ? dragKindOf(event) : null

      if (!kind) {
        return
      }

      // An outer layer may have already claimed this drop via preventDefault —
      // reset the hover state but don't ALSO act on it.
      const claimed = event.defaultPrevented

      event.preventDefault()
      reset()

      if (claimed) {
        return
      }

      const files = extractDroppedFiles(event.dataTransfer)

      if (files.length) {
        onDropFiles(files)
      }
    },
    [enabled, onDropFiles, reset]
  )

  return {
    dragKind,
    dropHandlers: { onDragEnter, onDragLeave, onDragOver, onDrop, onDropCapture: reset }
  }
}

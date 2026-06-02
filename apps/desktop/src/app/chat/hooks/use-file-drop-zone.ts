import { type DragEvent as ReactDragEvent, useCallback, useRef, useState } from 'react'

import { dragHasAttachments } from '@/app/chat/composer/inline-refs'

import { type DroppedFile, extractDroppedFiles, HERMES_PATHS_MIME } from './use-composer-actions'

const hasFiles = (event: ReactDragEvent) => dragHasAttachments(event.dataTransfer, HERMES_PATHS_MIME)

interface FileDropZoneOptions {
  /** When false the zone ignores drags entirely. */
  enabled?: boolean
  onDropFiles: (files: DroppedFile[]) => void
}

/**
 * "Drop files anywhere in this region" affordance. An enter/leave depth counter
 * keeps nested children from flickering the active state; `onDropCapture` clears
 * it even when a nested target (the composer) handles the drop and stops
 * propagation before our bubble-phase `onDrop` would fire.
 *
 * Spread `dropHandlers` onto the container; render an overlay off `dragActive`.
 */
export function useFileDropZone({ enabled = true, onDropFiles }: FileDropZoneOptions) {
  const [dragActive, setDragActive] = useState(false)
  const depth = useRef(0)

  const reset = useCallback(() => {
    depth.current = 0
    setDragActive(false)
  }, [])

  const onDragEnter = useCallback(
    (event: ReactDragEvent) => {
      if (!enabled || !hasFiles(event)) {
        return
      }

      event.preventDefault()
      depth.current += 1
      setDragActive(true)
    },
    [enabled]
  )

  const onDragOver = useCallback(
    (event: ReactDragEvent) => {
      if (!enabled || !hasFiles(event)) {
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
      if (!enabled || !hasFiles(event)) {
        return
      }

      event.preventDefault()
      reset()

      const files = extractDroppedFiles(event.dataTransfer)

      if (files.length) {
        onDropFiles(files)
      }
    },
    [enabled, onDropFiles, reset]
  )

  return { dragActive, dropHandlers: { onDragEnter, onDragLeave, onDragOver, onDrop, onDropCapture: reset } }
}

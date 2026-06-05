import { type DragEvent as ReactDragEvent, useCallback, useRef, useState } from 'react'

import {
  dragHasAttachments,
  dragHasSession,
  readSessionDrag,
  type SessionDragPayload
} from '@/app/chat/composer/inline-refs'

import { type DroppedFile, extractDroppedFiles, HERMES_PATHS_MIME } from './use-composer-actions'

export type DragKind = 'files' | 'session' | null

const dragKindOf = (event: ReactDragEvent): DragKind => {
  if (dragHasSession(event.dataTransfer)) {
    return 'session'
  }

  if (dragHasAttachments(event.dataTransfer, HERMES_PATHS_MIME)) {
    return 'files'
  }

  return null
}

interface FileDropZoneOptions {
  /** When false the zone ignores drags entirely. */
  enabled?: boolean
  onDropFiles: (files: DroppedFile[]) => void
  onDropSession?: (session: SessionDragPayload) => void
}

/**
 * "Drop anywhere in this region" affordance for files *and* in-app session
 * links. An enter/leave depth counter keeps nested children from flickering the
 * active state; `onDropCapture` clears it even when a nested target (the
 * composer) handles the drop and stops propagation before our bubble-phase
 * `onDrop` would fire.
 *
 * Spread `dropHandlers` onto the container; render an overlay off `dragKind`.
 */
export function useFileDropZone({ enabled = true, onDropFiles, onDropSession }: FileDropZoneOptions) {
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

      event.preventDefault()
      reset()

      if (kind === 'session') {
        const session = readSessionDrag(event.dataTransfer)

        if (session) {
          onDropSession?.(session)
        }

        return
      }

      const files = extractDroppedFiles(event.dataTransfer)

      if (files.length) {
        onDropFiles(files)
      }
    },
    [enabled, onDropFiles, onDropSession, reset]
  )

  return {
    dragKind,
    dropHandlers: { onDragEnter, onDragLeave, onDragOver, onDrop, onDropCapture: reset }
  }
}

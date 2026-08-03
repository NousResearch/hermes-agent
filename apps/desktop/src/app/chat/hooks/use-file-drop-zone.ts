import { type DragEvent as ReactDragEvent, useCallback, useEffect, useRef, useState } from 'react'

import { dragHasAttachments } from '@/app/chat/composer/inline-refs'
import { ESCAPE_PRIORITY, pushEscapeLayer } from '@/lib/escape-layers'

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
 * Esc aborts an in-flight drag, matching the sidebar session drag.
 */
export function useFileDropZone({ enabled = true, onDropFiles }: FileDropZoneOptions) {
  const [dragKind, setDragKind] = useState<DragKind>(null)
  const depth = useRef(0)
  const aborted = useRef(false)

  const reset = useCallback(() => {
    depth.current = 0
    setDragKind(null)
  }, [])

  // Esc aborts a file drag — the same "never mind" a session drag gets. Native
  // DnD can't be cancelled at the OS level, so we drop the overlay and arm a
  // guard that swallows the trailing drop instead. Top escape layer + capture
  // stop so it doesn't also fire a handler behind the drag (see drag-session).
  useEffect(() => {
    if (dragKind === null) {
      return
    }

    const releaseLayer = pushEscapeLayer(ESCAPE_PRIORITY.drag)

    const onKey = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') {
        return
      }

      event.preventDefault()
      event.stopPropagation()
      aborted.current = true
      reset()
    }

    window.addEventListener('keydown', onKey, true)

    return () => {
      window.removeEventListener('keydown', onKey, true)
      releaseLayer()
    }
  }, [dragKind, reset])

  const onDragEnter = useCallback(
    (event: ReactDragEvent) => {
      const kind = enabled ? dragKindOf(event) : null

      if (!kind) {
        return
      }

      event.preventDefault()

      // A genuinely new drag (not a nested-child re-enter) re-arms after abort.
      if (depth.current === 0) {
        aborted.current = false
      }

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

      // Only an Esc abort swallows the drop — NOT `event.defaultPrevented`. The
      // file tree's app-wide react-dnd HTML5Backend preventDefaults every native
      // file drop in the capture phase, so that flag is always set here (every
      // Finder drop would no-op). Genuine nested targets claim via stopPropagation
      // and never reach this bubble handler anyway.
      const claimed = aborted.current

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

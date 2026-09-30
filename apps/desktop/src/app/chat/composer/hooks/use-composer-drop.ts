import { type DragEvent as ReactDragEvent, useRef, useState } from 'react'

import { triggerHaptic } from '@/lib/haptics'
import { $terminalBackend } from '@/store/session'
import { isSessionRemote } from '@/store/session-states'

import {
  extractDroppedFiles,
  HERMES_PATHS_MIME,
  type OsDropStagingContext,
  partitionDroppedFiles
} from '../../hooks/use-composer-actions'
import { dragHasAttachments, droppedFileInlineRefs, type InlineRefInput } from '../inline-refs'
import type { ChatBarProps } from '../types'

interface UseComposerDropArgs {
  cwd: ChatBarProps['cwd']
  insertInlineRefs: (refs: InlineRefInput[]) => boolean
  onAttachDroppedItems: ChatBarProps['onAttachDroppedItems']
  recordUndoPoint: () => void
  requestMainFocus: () => void
  sessionId?: string | null
}

/**
 * Drag-and-drop attachment engine. Splits drops by origin: in-app drags
 * (project tree / gutter) stay inline `@file:`/`@line:` refs the gateway
 * resolves directly; OS/Finder drops (absolute local paths a remote gateway
 * can't read, image bytes vision needs) route through the upload pipeline.
 * Off the keystroke path; consumes `insertInlineRefs` + the attach handler.
 */
export function useComposerDrop({
  cwd,
  insertInlineRefs,
  onAttachDroppedItems,
  recordUndoPoint,
  requestMainFocus,
  sessionId
}: UseComposerDropArgs) {
  const [dragActive, setDragActive] = useState(false)
  const dragDepthRef = useRef(0)

  const resetDragState = () => {
    dragDepthRef.current = 0
    setDragActive(false)
  }

  // Staging inputs for OS drops, read at drop time (#52427): a local
  // connection on a shared filesystem keeps non-image OS drops as
  // original-path inline refs; remote / container / cross-filesystem
  // backends still stage. Mirrors uploadComposerAttachment's byte decision.
  const osDropStaging = (): OsDropStagingContext => ({
    backendCwd: cwd,
    remote: isSessionRemote(sessionId),
    terminalBackend: $terminalBackend.get()
  })

  const handleDragEnter = (event: ReactDragEvent<HTMLFormElement>) => {
    if (!onAttachDroppedItems || !dragHasAttachments(event.dataTransfer, HERMES_PATHS_MIME)) {
      return
    }

    event.preventDefault()
    dragDepthRef.current += 1

    if (!dragActive) {
      setDragActive(true)
    }
  }

  const handleDragOver = (event: ReactDragEvent<HTMLFormElement>) => {
    if (!onAttachDroppedItems || !dragHasAttachments(event.dataTransfer, HERMES_PATHS_MIME)) {
      return
    }

    event.preventDefault()
    event.dataTransfer.dropEffect = 'copy'
  }

  const handleDragLeave = (event: ReactDragEvent<HTMLFormElement>) => {
    if (!onAttachDroppedItems) {
      return
    }

    event.preventDefault()
    dragDepthRef.current = Math.max(0, dragDepthRef.current - 1)

    if (dragDepthRef.current === 0) {
      setDragActive(false)
    }
  }

  const handleDrop = (event: ReactDragEvent<HTMLFormElement>) => {
    if (!onAttachDroppedItems) {
      return
    }

    event.preventDefault()
    resetDragState()

    const candidates = extractDroppedFiles(event.dataTransfer)

    if (candidates.length === 0) {
      return
    }

    // In-app drags (project tree / gutter) are workspace-relative paths the
    // gateway resolves directly, so they stay inline @file:/@line: refs. OS
    // drops are absolute local paths a remote gateway can't read (and images
    // need byte upload for vision), so route them through the upload pipeline
    // — unless the backend resolves this machine's paths as-is (#52427).
    const { inAppRefs, osDrops } = partitionDroppedFiles(candidates, osDropStaging())
    const refs = droppedFileInlineRefs(inAppRefs, cwd)

    if (refs.length && insertInlineRefs(refs)) {
      triggerHaptic('selection')
    }

    if (osDrops.length) {
      void Promise.resolve(onAttachDroppedItems(osDrops)).then(attached => {
        if (attached) {
          triggerHaptic('selection')
          requestMainFocus()
        }
      })
    }
  }

  const handleInputDragOver = (event: ReactDragEvent<HTMLDivElement>) => {
    if (!dragHasAttachments(event.dataTransfer, HERMES_PATHS_MIME)) {
      return
    }

    event.preventDefault()
    event.stopPropagation()
    event.dataTransfer.dropEffect = 'copy'
  }

  const handleInputDrop = (event: ReactDragEvent<HTMLDivElement>) => {
    if (!dragHasAttachments(event.dataTransfer, HERMES_PATHS_MIME)) {
      // A plain text drag within the editor mutates the DOM without a
      // React-visible beforeinput (insertFromDrop), so the undo snapshot has
      // to be banked here — before Chromium applies the move.
      recordUndoPoint()

      return
    }

    const candidates = extractDroppedFiles(event.dataTransfer)

    if (!candidates.length) {
      return
    }

    event.preventDefault()
    event.stopPropagation()
    resetDragState()

    // Dropping straight onto the text box used to inline-ref *every* file —
    // including OS/Finder drops, whose absolute local path a remote gateway
    // can't read and whose image bytes never reached vision. Split by origin:
    // in-app drags stay inline refs; OS drops go through the upload pipeline.
    // (When no upload handler is wired, fall back to inline refs for all.)
    const attach = onAttachDroppedItems
    const { inAppRefs, osDrops } = partitionDroppedFiles(candidates, osDropStaging())
    const refs = droppedFileInlineRefs(attach ? inAppRefs : candidates, cwd)

    if (refs.length && insertInlineRefs(refs)) {
      triggerHaptic('selection')
    }

    if (attach && osDrops.length) {
      void Promise.resolve(attach(osDrops)).then(attached => {
        if (attached) {
          triggerHaptic('selection')
          requestMainFocus()
        }
      })
    }
  }

  // OS file-paste entry point (#128823). A file-manager copy (Finder /
  // Explorer) pasted into the composer delivers native File entries on the
  // paste event's clipboardData alongside a text form (filenames / file
  // URLs). Inserting that text form garbles the composer and attaches
  // nothing, so non-image file payloads take the same split as drops:
  // in-app refs stay inline, OS drops go through the upload pipeline (whose
  // failures already surface as notices). Image files stay out — the paste
  // handler's image-blob path owns those. Returns true when a file payload
  // was consumed and the caller must skip the text form.
  const handlePasteFiles = (clipboard: DataTransfer): boolean => {
    if (!onAttachDroppedItems) {
      return false
    }

    const candidates = extractDroppedFiles(clipboard).filter(
      candidate => candidate.file && !candidate.file.type.startsWith('image/')
    )

    if (candidates.length === 0) {
      return false
    }

    const { inAppRefs, osDrops } = partitionDroppedFiles(candidates, osDropStaging())
    const refs = droppedFileInlineRefs(inAppRefs, cwd)

    if (refs.length && insertInlineRefs(refs)) {
      triggerHaptic('selection')
    }

    if (osDrops.length) {
      void Promise.resolve(onAttachDroppedItems(osDrops)).then(attached => {
        if (attached) {
          triggerHaptic('selection')
          requestMainFocus()
        }
      })
    }

    return true
  }

  return {
    dragActive,
    handleDragEnter,
    handleDragLeave,
    handleDragOver,
    handleDrop,
    handleInputDragOver,
    handleInputDrop,
    handlePasteFiles
  }
}

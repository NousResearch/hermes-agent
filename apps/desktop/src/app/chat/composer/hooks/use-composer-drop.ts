import { type DragEvent as ReactDragEvent, useRef, useState } from 'react'

import { triggerHaptic } from '@/lib/haptics'

import { extractDroppedFiles, HERMES_PATHS_MIME, partitionDroppedFiles } from '../../hooks/use-composer-actions'
import { createDragLifecycleGate, type DragLifecycleGate, droppedFileInlineRefs, type InlineRefInput } from '../inline-refs'
import type { ChatBarProps } from '../types'

interface UseComposerDropArgs {
  cwd: ChatBarProps['cwd']
  insertInlineRefs: (refs: InlineRefInput[]) => boolean
  onAttachDroppedItems: ChatBarProps['onAttachDroppedItems']
  requestMainFocus: () => void
}

/**
 * Drag-and-drop attachment engine. Splits drops by origin: in-app drags
 * (project tree / gutter) stay inline `@file:`/`@line:` refs the gateway
 * resolves directly; OS/Finder drops (absolute local paths a remote gateway
 * can't read, image bytes vision needs) route through the upload pipeline.
 * Off the keystroke path; consumes `insertInlineRefs` + the attach handler.
 *
 * Windows sparse drag (#97702): the shared `createDragLifecycleGate`
 * arms on the first dragenter signal (typed file drag OR a fully-empty
 * Windows Explorer transfer) and keeps dragover alive for the same drag,
 * so the app-lifetime react-dnd HTML5Backend's capture-time
 * `dropEffect='none'` cannot suppress the eventual drop.
 */
export function useComposerDrop({
  cwd,
  insertInlineRefs,
  onAttachDroppedItems,
  requestMainFocus
}: UseComposerDropArgs) {
  const [dragActive, setDragActive] = useState(false)
  const dragDepthRef = useRef(0)
  // One gate per composer instance; stable across re-renders.
  const gateRef = useRef<DragLifecycleGate | null>(null)
  const gate = gateRef.current ?? (gateRef.current = createDragLifecycleGate(HERMES_PATHS_MIME))

  const resetDragState = () => {
    dragDepthRef.current = 0
    gate.reset()
    setDragActive(false)
  }

  const handleDragEnter = (event: ReactDragEvent<HTMLFormElement>) => {
    // A genuinely new drag (not a nested-child re-enter) must reset
    // any unrecovered gate state before the new drag is evaluated.
    if (dragDepthRef.current === 0) {
      gate.reset()
    }

    if (!onAttachDroppedItems || !gate.onEnter(event.dataTransfer)) {
      return
    }

    event.preventDefault()
    dragDepthRef.current += 1

    if (!dragActive) {
      setDragActive(true)
    }
  }

  const handleDragOver = (event: ReactDragEvent<HTMLFormElement>) => {
    if (!onAttachDroppedItems || !gate.onOver(event.dataTransfer)) {
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
      gate.onLeave(true)
      setDragActive(false)
    } else {
      gate.onLeave(false)
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
    // need byte upload for vision), so route them through the upload pipeline.
    const { inAppRefs, osDrops } = partitionDroppedFiles(candidates)
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
    if (!gate.onOver(event.dataTransfer)) {
      return
    }

    event.preventDefault()
    event.stopPropagation()
    event.dataTransfer.dropEffect = 'copy'
  }

  const handleInputDrop = (event: ReactDragEvent<HTMLDivElement>) => {
    if (!gate.onOver(event.dataTransfer)) {
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
    const { inAppRefs, osDrops } = partitionDroppedFiles(candidates)
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

  return {
    dragActive,
    handleDragEnter,
    handleDragLeave,
    handleDragOver,
    handleDrop,
    handleInputDragOver,
    handleInputDrop
  }
}

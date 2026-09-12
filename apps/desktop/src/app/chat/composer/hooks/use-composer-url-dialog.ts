import { useEffect, useRef, useState } from 'react'

import { triggerHaptic } from '@/lib/haptics'

interface UseComposerUrlDialogOptions {
  /** Insert a new `@url:` directive into the draft (attach mode submit). */
  insertText: (text: string) => void
  onAddUrl?: (url: string) => void
  /** Rewrite a reference chip in place. Returns false when the chip vanished
   *  (deleted/undone) before the edit landed — the caller keeps the dialog
   *  open so the user can retry or cancel. */
  applyChipEdit: (chip: HTMLElement, value: string) => boolean
  /** Bank the pre-edit state on the composer's undo stack before a chip edit. */
  recordChipEditUndo: () => void
  /** Push the live editor text back into draft + composer state after an edit. */
  syncEditorToDraft: () => void
}

interface ChipEdit {
  chip: HTMLElement
  value: string
}

/**
 * The URL dialog engine: open/value state, autofocus-on-open, and submit.
 *
 * Two modes share the one dialog:
 *  - **attach** (the "Add URL" action) — submit prefers the host's `onAddUrl`
 *    (which may fetch/title the link) and otherwise drops an `@url:` directive
 *    into the draft.
 *  - **chip edit** (double-clicking a reference chip) — the field is pre-filled
 *    with the chip's current value and submit rewrites that chip in place,
 *    which is the only way a `contenteditable="false"` chip can be corrected
 *    mid-draft (it is atomic: the caret cannot enter it).
 */
export function useComposerUrlDialog({
  applyChipEdit,
  insertText,
  onAddUrl,
  recordChipEditUndo,
  syncEditorToDraft
}: UseComposerUrlDialogOptions) {
  const urlInputRef = useRef<HTMLInputElement | null>(null)
  const [urlOpen, setUrlOpen] = useState(false)
  const [urlValue, setUrlValue] = useState('')
  const [chipEdit, setChipEdit] = useState<ChipEdit | null>(null)

  useEffect(() => {
    if (urlOpen) {
      window.requestAnimationFrame(() => urlInputRef.current?.focus({ preventScroll: true }))
    }
  }, [urlOpen])

  const openUrlDialog = () => {
    triggerHaptic('open')
    setChipEdit(null)
    setUrlValue('')
    setUrlOpen(true)
  }

  /** Double-click a reference chip: open the dialog pre-filled with its value,
   *  in edit mode. */
  const beginChipEdit = (chip: HTMLElement, value: string) => {
    if (urlOpen) {
      // A dialog is already open (attach or a prior edit) — don't stack a
      // second one on it.
      return
    }

    triggerHaptic('open')
    setChipEdit({ chip, value })
    setUrlValue(value)
    setUrlOpen(true)
  }

  const closeUrlDialog = () => {
    setChipEdit(null)
    setUrlOpen(false)
  }

  const submitUrl = () => {
    const url = urlValue.trim()

    if (!url) {
      return
    }

    if (chipEdit) {
      // Editing an existing chip: bank the pre-edit state, rewrite in place,
      // and pull the live DOM text back into draft + composer state so the
      // submit button and queue reflect the corrected reference.
      recordChipEditUndo()

      if (!applyChipEdit(chipEdit.chip, url)) {
        // The chip is gone (deleted/undone while the dialog was open) — keep
        // the dialog open on the value so the user can attach it instead or
        // cancel; closing would silently discard their edit.
        triggerHaptic('selection')

        return
      }

      syncEditorToDraft()
      triggerHaptic('success')
      setChipEdit(null)
      setUrlValue('')
      setUrlOpen(false)

      return
    }

    if (onAddUrl) {
      onAddUrl(url)
    } else {
      insertText(`@url:${url}`)
    }

    triggerHaptic('success')
    setUrlValue('')
    setUrlOpen(false)
  }

  return {
    beginChipEdit,
    chipEdit,
    closeUrlDialog,
    openUrlDialog,
    setUrlOpen: closeUrlDialog,
    setUrlValue,
    submitUrl,
    urlInputRef,
    urlOpen,
    urlValue
  }
}

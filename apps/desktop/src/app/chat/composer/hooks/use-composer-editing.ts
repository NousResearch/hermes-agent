import { type RefObject, useCallback, useEffect, useLayoutEffect, useRef } from 'react'

import { triggerHaptic } from '@/lib/haptics'
import type { ComposerAttachment } from '@/store/composer'

import { insertAttachmentReference, refreshAttachmentReferences } from '../attachment-reference-editor'
import { attachmentReferenceName, setAttachmentNamesForEditor } from '../attachment-references'
import { onComposerAttachImagesRequest } from '../focus'
import { useComposerScope } from '../scope'
import type { ChatBarProps } from '../types'

import { useComposerUndo } from './use-composer-undo'

interface ComposerEditingArgs {
  onAttachImageBlob: ChatBarProps['onAttachImageBlob']
  editorRef: RefObject<HTMLDivElement | null>
  composingRef: RefObject<boolean>
  sessionKey: string | null
  syncDraftFromEditor: () => string
}

export function useComposerEditing({
  onAttachImageBlob,
  editorRef,
  composingRef,
  sessionKey,
  syncDraftFromEditor
}: ComposerEditingArgs) {
  const { attachments, target: composerTarget } = useComposerScope()
  const pending = useRef<ComposerAttachment[]>([])
  const undo = useComposerUndo({ editorRef, syncDraftFromEditor })
  const { recordUndoPoint, resetUndoHistory } = undo

  // Restored drafts cannot undo into another conversation's text.
  useEffect(() => {
    resetUndoHistory()
  }, [sessionKey, resetUndoHistory])

  const insert = useCallback(
    (attachment: ComposerAttachment) => {
      const editor = editorRef.current
      const name = attachmentReferenceName(attachment)

      if (editor && name) {
        recordUndoPoint()
        insertAttachmentReference(editor, name)
        syncDraftFromEditor()
      }
    },
    [editorRef, recordUndoPoint, syncDraftFromEditor]
  )

  const refresh = useCallback(() => {
    const editor = editorRef.current

    if (!editor || composingRef.current) {
      return
    }

    const current = attachments.$attachments.get()
    setAttachmentNamesForEditor(
      editor,
      current.flatMap(attachment => attachmentReferenceName(attachment) ?? [])
    )
    refreshAttachmentReferences(editor)

    for (const attachment of pending.current.splice(0)) {
      if (current.some(item => item.id === attachment.id)) {
        insert(attachment)
      }
    }
  }, [attachments, composingRef, editorRef, insert])

  useLayoutEffect(() => {
    pending.current = []
    const offNames = attachments.$attachments.subscribe(refresh)

    const offAdd = attachments.onAdd(attachment => {
      if (composingRef.current) {
        pending.current.push(attachment)
      } else {
        insert(attachment)
      }
    })

    return () => {
      offNames()
      offAdd()
      pending.current = []
    }
  }, [attachments, composingRef, insert, refresh, sessionKey])

  // Paste-to-focus: clipboard images from an unfocused ⌘V ride the bus (the
  // window dispatcher has no handle on this composer's attachment scope).
  // Same ingestion as a focused paste's image branch.
  useEffect(() => {
    if (!onAttachImageBlob) {
      return undefined
    }

    return onComposerAttachImagesRequest(({ blobs, target }) => {
      if (target !== composerTarget) {
        return
      }

      triggerHaptic('selection')

      for (const blob of blobs) {
        void onAttachImageBlob(blob)
      }
    })
  }, [onAttachImageBlob, composerTarget])

  return { ...undo, refreshAttachmentReferences: refresh }
}

import { type RefObject, useRef } from 'react'

import { isElementInHiddenPane } from '@/components/pane-shell/pane-visibility'
import { useI18n } from '@/i18n'
import type { ComposerAttachment } from '@/store/composer'
import { notifyError } from '@/store/notifications'

import { composerPlainText } from '../rich-editor'
import { useComposerScope } from '../scope'

interface RestorePastedTextArgs {
  activeQueueSessionKeyRef: RefObject<string | null>
  editorRef: RefObject<HTMLDivElement | null>
  inputDisabled: boolean
  loadIntoComposer: (text: string, attachments: ComposerAttachment[]) => void
  focusInput: () => void
}

/** Restore a saved paste only into the draft/attachment occurrence that requested it. */
export function useRestorePastedText({
  activeQueueSessionKeyRef,
  editorRef,
  inputDisabled,
  loadIntoComposer,
  focusInput
}: RestorePastedTextArgs) {
  const { t } = useI18n()
  const { attachments } = useComposerScope()
  const disabledRef = useRef(inputDisabled)
  disabledRef.current = inputDisabled
  const pending = useRef(new Set<ComposerAttachment>())

  return async (attachment: ComposerAttachment) => {
    const editor = editorRef.current
    const sessionKey = activeQueueSessionKeyRef.current
    const path = attachment.pastedTextPath || attachment.path

    if (!editor || !path || disabledRef.current || pending.current.has(attachment)) {
      return
    }

    pending.current.add(attachment)

    try {
      // savePastedText writes on the Desktop client, even for a remote gateway.
      // Never resolve its path against the agent's filesystem or use titlePreview.
      const result = await window.hermesDesktop?.readFileText(path)

      if (!result || result.binary || result.truncated || !result.text.trim()) {
        throw new Error(t.composer.pasteTextIncomplete)
      }

      const current = attachments.$attachments.get()

      const index = current.findIndex(item =>
        attachment.occurrenceId === undefined
          ? item === attachment
          : item.id === attachment.id && item.occurrenceId === attachment.occurrenceId
      )

      if (
        index < 0 ||
        activeQueueSessionKeyRef.current !== sessionKey ||
        editorRef.current !== editor ||
        !editor.isConnected ||
        isElementInHiddenPane(editor) ||
        disabledRef.current
      ) {
        return
      }

      // Read after the await: edits made while the file was loading must survive.
      const draft = composerPlainText(editor)
      const separator = draft && !draft.endsWith('\n') ? '\n\n' : ''
      loadIntoComposer(
        `${draft}${separator}${result.text}`,
        current.filter((_, i) => i !== index)
      )
      focusInput()
    } catch (error) {
      notifyError(error, t.composer.pasteTextRestoreFailed)
    } finally {
      pending.current.delete(attachment)
    }
  }
}

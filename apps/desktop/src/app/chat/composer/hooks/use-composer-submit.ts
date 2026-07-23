import { type RefObject, useEffect, useRef } from 'react'

import { SLASH_COMMAND_RE } from '@/lib/chat-runtime'
import { triggerHaptic } from '@/lib/haptics'
import { clearSessionDraft, type ComposerAttachment } from '@/store/composer'
import { resetBrowseState } from '@/store/composer-input-history'
import { enqueueQueuedPrompt, type QueuedPromptEntry } from '@/store/composer-queue'

import type { BusyInputMode } from '../busy-input-mode'
import { cloneAttachments, type QueueEditState } from '../composer-utils'
import { onComposerSubmitRequest } from '../focus'
import { composerPlainText } from '../rich-editor'
import { useComposerScope } from '../scope'
import type { ChatBarProps } from '../types'

interface UseComposerSubmitArgs {
  activeQueueSessionKey: string | null
  activeQueueSessionKeyRef: RefObject<string | null>
  attachments: ComposerAttachment[]
  busy: boolean
  busyInputMode: BusyInputMode
  compacting: boolean
  clearDraft: () => void
  disabled: boolean
  draftRef: RefObject<string>
  drainNextQueued: () => Promise<boolean>
  editorRef: RefObject<HTMLDivElement | null>
  exitQueuedEdit: (action: 'cancel' | 'save') => boolean
  focusInput: () => void
  inputDisabled: boolean
  loadIntoComposer: (text: string, attachments: ComposerAttachment[]) => void
  onCancel: ChatBarProps['onCancel']
  onSteer: ChatBarProps['onSteer']
  onSubmit: ChatBarProps['onSubmit']
  onToolSteer: ChatBarProps['onToolSteer']
  queueCurrentDraft: () => boolean
  queueEdit: QueueEditState | null
  queuedPrompts: QueuedPromptEntry[]
  sessionId: string | null | undefined
  setComposerText: (value: string) => void
  stashAt: (scope: string | null, text?: string, attachments?: ComposerAttachment[]) => void
}

/**
 * The composer's submit engine — the orchestration seam where the draft and
 * queue meet. `submitDraft` is the one decision tree (queue-edit save · slash-
 * now-while-busy · queue · drain · send · stop); `dispatchSubmit` is the shared
 * send-with-restore primitive (re-loads + re-stashes the draft if the gateway
 * rejects, so nothing is ever lost).
 */
export function useComposerSubmit({
  activeQueueSessionKey,
  activeQueueSessionKeyRef,
  attachments,
  busy,
  busyInputMode,
  compacting,
  clearDraft,
  disabled,
  draftRef,
  drainNextQueued,
  editorRef,
  exitQueuedEdit,
  focusInput,
  inputDisabled,
  loadIntoComposer,
  onCancel,
  onSteer,
  onSubmit,
  onToolSteer,
  queueCurrentDraft,
  queueEdit,
  queuedPrompts,
  sessionId,
  setComposerText,
  stashAt
}: UseComposerSubmitArgs) {
  const scope = useComposerScope()

  // Shared send primitive: fire onSubmit, and if the gateway rejects (accepted
  // === false) or throws, re-load + re-stash the draft so the words survive.
  const dispatchSubmit = (text: string, attachments?: ComposerAttachment[]) => {
    const submittedScope = activeQueueSessionKeyRef.current
    const submittedAttachments = attachments ?? []

    const restore = () => {
      loadIntoComposer(text, submittedAttachments)
      // Use the scope captured at dispatch, not whatever session is focused
      // now — the gateway can reject well after the user has switched away,
      // and re-stashing into the currently-focused session would overwrite
      // its draft with the rejected text from a different session (#54527).
      stashAt(submittedScope, text, submittedAttachments)
    }

    void Promise.resolve(attachments ? onSubmit(text, { attachments }) : onSubmit(text))
      .then(accepted => void (accepted === false ? restore() : clearSessionDraft(submittedScope)))
      .catch(restore)
  }

  // External "submit this prompt" requests (e.g. the review pane's agent-ship
  // button) route through the same send path. A ref keeps the listener stable
  // while always calling the latest dispatchSubmit closure.
  const dispatchSubmitRef = useRef(dispatchSubmit)
  dispatchSubmitRef.current = dispatchSubmit

  useEffect(
    () =>
      onComposerSubmitRequest(({ target, text }) => {
        if (target === 'main' && !inputDisabled) {
          dispatchSubmitRef.current(text)
        }
      }),
    [inputDisabled]
  )

  const queueFallback = (text: string) => {
    if (activeQueueSessionKey) {
      enqueueQueuedPrompt(activeQueueSessionKey, { text, attachments: [] })
    }
  }

  const redirectDraft = () => {
    const text = draftRef.current.trim()

    if (!onSteer || !text || attachments.length > 0 || SLASH_COMMAND_RE.test(text)) {
      queueCurrentDraft()
      return
    }

    triggerHaptic('submit')
    clearDraft()

    void Promise.resolve(onSteer(text)).then(accepted => {
      if (!accepted) {
        queueFallback(text)
      }
    })
  }

  const toolSteerDraft = () => {
    const text = draftRef.current.trim()

    if (!onToolSteer || !text || attachments.length > 0 || SLASH_COMMAND_RE.test(text)) {
      queueCurrentDraft()
      return
    }

    triggerHaptic('submit')
    clearDraft()

    void Promise.resolve(onToolSteer(text)).then(accepted => {
      if (!accepted) {
        queueFallback(text)
      }
    })
  }

  const submitDraft = () => {
    if (disabled) {
      return
    }

    // Source the text from the DOM editor, not React state. The AUI composer
    // state (`draft`) and the derived `hasComposerPayload` lag the DOM by a
    // render, so on fast typing or IME composition the final keystroke(s) may
    // not have synced yet — reading state here drops the message (Enter looks
    // like it does nothing; typing a trailing space only "fixes" it because the
    // extra input event forces a state sync). draftRef is updated on every
    // input event; refresh it from the editor once more to also cover an
    // in-flight keystroke that hasn't fired its input event yet.
    const editor = editorRef.current

    if (editor) {
      const domText = composerPlainText(editor)

      if (domText !== draftRef.current) {
        draftRef.current = domText
        setComposerText(domText)
      }
    }

    const text = draftRef.current
    const payloadPresent = text.trim().length > 0 || attachments.length > 0

    if (queueEdit) {
      exitQueuedEdit('save')
    } else if (busy) {
      // Slash commands execute immediately even while the agent is busy.
      if (!attachments.length && SLASH_COMMAND_RE.test(text.trim())) {
        triggerHaptic('submit')
        clearDraft()
        dispatchSubmit(text)
      } else if (!compacting && !attachments.length && text.trim()) {
        if (busyInputMode === 'queue') {
          queueCurrentDraft()
        } else if (busyInputMode === 'steer') {
          toolSteerDraft()
        } else {
          redirectDraft()
        }
      } else if (payloadPresent) {
        // Attachments cannot ride either redirect or tool-boundary steering.
        queueCurrentDraft()
      } else {
        triggerHaptic('cancel')
        void Promise.resolve(onCancel())
      }
    } else if (!payloadPresent && queuedPrompts.length > 0) {
      void drainNextQueued()
    } else if (payloadPresent) {
      const submittedAttachments = cloneAttachments(attachments)
      triggerHaptic('submit')
      resetBrowseState(sessionId)
      clearDraft()
      scope.attachments.clear()
      dispatchSubmit(text, submittedAttachments)
    }

    focusInput()
  }

  const queueDraft = () => {
    if (disabled || !busy) {
      return
    }

    queueCurrentDraft()
    focusInput()
  }

  return { dispatchSubmit, queueDraft, redirectDraft, toolSteerDraft, submitDraft }
}

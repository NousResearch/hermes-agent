import { type RefObject, useCallback, useEffect, useRef, useState } from 'react'

import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { useSessionSlice } from '@/lib/use-session-slice'
import { clearComposerAttachments, type ComposerAttachment } from '@/store/composer'
import { resetBrowseState } from '@/store/composer-input-history'
import {
  $queuedPromptsBySession,
  enqueueQueuedPrompt,
  markQueuedPromptAutoDrain,
  markQueuedPromptsAutoDrain,
  MAX_AUTO_DRAIN_ATTEMPTS,
  migrateQueuedPrompts,
  promoteQueuedPrompt,
  type QueuedPromptEntry,
  removeQueuedPrompt,
  shouldAutoDrain,
  updateQueuedPrompt
} from '@/store/composer-queue'
import { notify } from '@/store/notifications'

import { cloneAttachments, type QueueEditState } from '../composer-utils'
import type { ChatBarProps } from '../types'

interface UseComposerQueueArgs {
  activeQueueSessionKey: string | null
  attachments: ComposerAttachment[]
  busy: boolean
  clearDraft: () => void
  draftRef: RefObject<string>
  focusInput: () => void
  loadIntoComposer: (text: string, attachments: ComposerAttachment[]) => void
  onCancel: ChatBarProps['onCancel']
  onSubmit: ChatBarProps['onSubmit']
  onDraftQueued?: () => void
  queueEditRef: RefObject<QueueEditState | null>
  queueSessionKey: ChatBarProps['queueSessionKey']
  sendBlocked: boolean
  sessionId: string | null | undefined
  transformDraftText?: (text: string) => string
}

/**
 * The composer's queue engine — everything about queued turns: the per-session
 * queue store binding, in-place queued-prompt editing (begin/step/exit), the
 * shared drain lock + send-then-remove sequence, manual send-now, and the
 * edge-independent auto-drain with bounded retries. It consumes the draft API
 * (draftRef/clearDraft/loadIntoComposer/focusInput) and writes the
 * coordinator-owned `queueEditRef` so the draft engine can read the edit state
 * without a back-reference. Behaviour-identical to the inline original.
 */
export function useComposerQueue({
  activeQueueSessionKey,
  attachments,
  busy,
  clearDraft,
  draftRef,
  focusInput,
  loadIntoComposer,
  onCancel,
  onSubmit,
  onDraftQueued,
  queueEditRef,
  queueSessionKey,
  sendBlocked,
  sessionId,
  transformDraftText
}: UseComposerQueueArgs) {
  const { t } = useI18n()

  // Per-session slice (edge): re-renders only when THIS session's queue changes,
  // not on cross-session queue churn (the plain atom's map ref changes on every
  // write; the keyed array does not).
  const queuedPrompts = useSessionSlice($queuedPromptsBySession, activeQueueSessionKey)

  const [queueEdit, setQueueEdit] = useState<QueueEditState | null>(null)
  queueEditRef.current = queueEdit

  const setQueueEditSnapshot = useCallback(
    (next: QueueEditState | null) => {
      queueEditRef.current = next
      setQueueEdit(next)
    },
    [queueEditRef]
  )

  const editingQueuedPrompt = queueEdit ? (queuedPrompts.find(entry => entry.id === queueEdit.entryId) ?? null) : null

  const prevQueueKeyRef = useRef(activeQueueSessionKey)
  const drainingQueueRef = useRef(false)
  const drainFailuresRef = useRef(new Map<string, number>())

  const beginQueuedEdit = (entry: QueuedPromptEntry) => {
    if (!activeQueueSessionKey || queueEdit) {
      return
    }

    setQueueEditSnapshot({
      attachments: cloneAttachments(attachments),
      draft: draftRef.current,
      entryId: entry.id,
      sessionKey: activeQueueSessionKey
    })
    loadIntoComposer(entry.text, entry.attachments)
    triggerHaptic('selection')
    focusInput()
  }

  // Walk queued entries while editing (ArrowUp = older, ArrowDown = newer),
  // saving the in-progress edit on each step. Stepping newer past the last
  // entry exits edit mode and restores the pre-edit draft.
  const stepQueuedEdit = (direction: -1 | 1) => {
    if (!queueEdit) {
      return false
    }

    const index = queuedPrompts.findIndex(e => e.id === queueEdit.entryId)
    const target = index + direction

    if (index < 0 || target < 0) {
      return index >= 0 // at the oldest: swallow; missing entry: let it fall through
    }

    const saved = updateQueuedPrompt(queueEdit.sessionKey, queueEdit.entryId, {
      attachments: cloneAttachments(attachments),
      text: draftRef.current
    })

    const next = queuedPrompts[target]

    if (next) {
      setQueueEditSnapshot({ ...queueEdit, entryId: next.id })
      loadIntoComposer(next.text, next.attachments)
    } else {
      setQueueEditSnapshot(null)
      loadIntoComposer(queueEdit.draft, queueEdit.attachments)
    }

    triggerHaptic(saved ? 'success' : 'selection')
    focusInput()

    return true
  }

  const exitQueuedEdit = (action: 'cancel' | 'save'): boolean => {
    if (!queueEdit) {
      return false
    }

    if (action === 'save') {
      const text = draftRef.current
      const next = cloneAttachments(attachments)

      if (!text.trim() && next.length === 0) {
        return false
      }

      const saved = updateQueuedPrompt(queueEdit.sessionKey, queueEdit.entryId, { attachments: next, text })
      triggerHaptic(saved ? 'success' : 'selection')
    } else {
      triggerHaptic('cancel')
    }

    setQueueEditSnapshot(null)
    loadIntoComposer(queueEdit.draft, queueEdit.attachments)
    focusInput()

    return true
  }

  const queueCurrentDraft = useCallback(() => {
    const text = draftRef.current

    if (!activeQueueSessionKey || (!text.trim() && attachments.length === 0)) {
      return false
    }

    if (
      !enqueueQueuedPrompt(activeQueueSessionKey, {
        text: transformDraftText?.(text) ?? text,
        attachments,
        autoDrain: sendBlocked
      })
    ) {
      return false
    }

    clearDraft()
    clearComposerAttachments()
    onDraftQueued?.()
    triggerHaptic('selection')

    return true
  }, [activeQueueSessionKey, attachments, clearDraft, draftRef, onDraftQueued, sendBlocked, transformDraftText])

  // All queue drain paths share one lock + send-then-remove sequence.
  // `pickEntry` lets each caller choose head, by-id, or skip-edited.
  const runDrain = useCallback(
    async (pickEntry: (entries: QueuedPromptEntry[]) => QueuedPromptEntry | undefined): Promise<boolean> => {
      if (drainingQueueRef.current || !activeQueueSessionKey) {
        return false
      }

      const entry = pickEntry(queuedPrompts)

      if (!entry) {
        return false
      }

      drainingQueueRef.current = true

      try {
        const accepted = await Promise.resolve(
          onSubmit(entry.text, { attachments: entry.attachments, fromQueue: true })
        )

        if (accepted === false) {
          return false
        }

        drainFailuresRef.current.delete(entry.id)
        removeQueuedPrompt(activeQueueSessionKey, entry.id)
        resetBrowseState(sessionId)

        return true
      } finally {
        drainingQueueRef.current = false
      }
    },
    [activeQueueSessionKey, onSubmit, queuedPrompts, sessionId]
  )

  const pickDrainHead = useCallback(
    (entries: QueuedPromptEntry[]) => {
      const skip = queueEditRef.current?.entryId

      return skip ? entries.find(e => e.id !== skip) : entries[0]
    },
    [queueEditRef] // reads the edit id off a ref so the lock-holder always sees the latest
  )

  const drainNextQueued = useCallback(() => runDrain(pickDrainHead), [pickDrainHead, runDrain])

  const sendQueuedNow = useCallback(
    (id: string) => {
      if (!activeQueueSessionKey || id === queueEdit?.entryId) {
        return false
      }

      if (sendBlocked) {
        // Promote to the head, then interrupt. The gateway always emits a
        // settle (message.complete + session.info running:false) when the
        // turn unwinds, and the busy→false auto-drain below sends this entry.
        promoteQueuedPrompt(activeQueueSessionKey, id)
        markQueuedPromptAutoDrain(activeQueueSessionKey, id)
        triggerHaptic('selection')

        if (busy) {
          void Promise.resolve(onCancel())
        }

        return true
      }

      // A manual send clears the auto-drain backoff so a stuck entry the user
      // taps gets a fresh attempt (and re-enables auto-retry on success).
      drainFailuresRef.current.delete(id)

      return runDrain(entries => entries.find(e => e.id === id))
    },
    [activeQueueSessionKey, busy, onCancel, queueEdit, runDrain, sendBlocked]
  )

  const sendAllQueued = useCallback(() => {
    if (!activeQueueSessionKey || queueEdit || queuedPrompts.length === 0) {
      return false
    }

    queuedPrompts.forEach(entry => drainFailuresRef.current.delete(entry.id))
    markQueuedPromptsAutoDrain(activeQueueSessionKey)

    if (sendBlocked) {
      triggerHaptic('selection')

      if (busy) {
        void Promise.resolve(onCancel())
      }

      return true
    }

    return drainNextQueued()
  }, [activeQueueSessionKey, busy, drainNextQueued, onCancel, queueEdit, queuedPrompts, sendBlocked])

  const implicitQueueDrainAllowed = useCallback(() => {
    const entry = pickDrainHead(queuedPrompts)

    return shouldAutoDrain({
      isBusy: sendBlocked,
      nextAutoDrain: Boolean(entry?.autoDrain),
      queueLength: queuedPrompts.length
    })
  }, [pickDrainHead, queuedPrompts, sendBlocked])

  // Edge-independent auto-drain: send the head whenever the session is idle and
  // the queue is non-empty, bounding retries so a thrown/rejected onSubmit (e.g.
  // a stale-session 404) can't strand the entry permanently nor spin-loop. The
  // drain lock serializes sends; a remount/reconnect resets the failure counts.
  const autoDrainNext = useCallback(() => {
    if (sendBlocked || drainingQueueRef.current || !activeQueueSessionKey) {
      return
    }

    const entry = pickDrainHead(queuedPrompts)

    if (!entry || (drainFailuresRef.current.get(entry.id) ?? 0) >= MAX_AUTO_DRAIN_ATTEMPTS) {
      return
    }

    const onFail = () => {
      const fails = (drainFailuresRef.current.get(entry.id) ?? 0) + 1
      drainFailuresRef.current.set(entry.id, fails)

      if (fails >= MAX_AUTO_DRAIN_ATTEMPTS) {
        notify({
          id: 'composer-queue-stuck',
          kind: 'error',
          title: t.composer.queueStuckTitle,
          message: t.composer.queueStuckBody
        })
      }
    }

    void runDrain(() => entry)
      .then(sent => {
        if (!sent) {
          onFail()
        }
      })
      .catch(onFail)
  }, [activeQueueSessionKey, pickDrainHead, queuedPrompts, runDrain, sendBlocked, t])

  // Re-key on a runtime session-id change. A stable stored id (queueSessionKey)
  // never churns, so a change there is a real session switch and must NOT
  // migrate; only the runtime-derived key (queueSessionKey falsy → key is
  // sessionId) churns on a backend bounce/resume of the same conversation.
  useEffect(() => {
    const prev = prevQueueKeyRef.current
    prevQueueKeyRef.current = activeQueueSessionKey

    if (queueSessionKey || !prev || !activeQueueSessionKey || prev === activeQueueSessionKey) {
      return
    }

    migrateQueuedPrompts(prev, activeQueueSessionKey)
  }, [activeQueueSessionKey, queueSessionKey])

  // Auto-drain only live follow-up turns. Older persisted/manual queue entries
  // stay visible until the user sends them, so reopening an idle session cannot
  // fire stale queue contents into the wrong context.
  useEffect(() => {
    if (implicitQueueDrainAllowed()) {
      autoDrainNext()
    }
  }, [autoDrainNext, implicitQueueDrainAllowed])

  // Queue-edit cleanup: on session swap the scope effect already stashed the
  // edit snapshot; only restore into the composer when still on the same scope.
  useEffect(() => {
    if (!queueEdit) {
      return
    }

    if (queueEdit.sessionKey === activeQueueSessionKey) {
      if (editingQueuedPrompt) {
        return
      }

      setQueueEditSnapshot(null)
      loadIntoComposer(queueEdit.draft, queueEdit.attachments)

      return
    }

    setQueueEditSnapshot(null)
  }, [activeQueueSessionKey, editingQueuedPrompt, queueEdit, setQueueEditSnapshot]) // eslint-disable-line react-hooks/exhaustive-deps

  return {
    beginQueuedEdit,
    drainNextQueued,
    editingQueuedPrompt,
    exitQueuedEdit,
    implicitQueueDrainAllowed,
    queueCurrentDraft,
    queueEdit,
    queuedPrompts,
    sendAllQueued,
    sendQueuedNow,
    stepQueuedEdit
  }
}

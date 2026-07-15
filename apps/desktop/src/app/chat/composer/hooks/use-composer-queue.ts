import { type RefObject, useCallback, useEffect, useRef, useState } from 'react'

import { triggerHaptic } from '@/lib/haptics'
import { useSessionSlice } from '@/lib/use-session-slice'
import { type ComposerAttachment } from '@/store/composer'
import {
  $pendingSteersBySession,
  $queuedPromptsBySession,
  migrateQueuedPrompts,
  promoteQueuedPrompt,
  type QueuedPromptEntry,
  updateQueuedPrompt
} from '@/store/composer-queue'

import { cloneAttachments, type QueueEditState } from '../composer-utils'
import { useComposerScope } from '../scope'
import type { ChatBarProps } from '../types'

interface UseComposerQueueArgs {
  activeQueueSessionKey: string | null
  attachments: ComposerAttachment[]
  busy: boolean
  clearDraft: () => void
  draftRef: RefObject<string>
  focusInput: () => void
  loadIntoComposer: (text: string, attachments: ComposerAttachment[]) => void
  onQueue: ChatBarProps['onQueue']
  queueEditRef: RefObject<QueueEditState | null>
  queueSessionKey: ChatBarProps['queueSessionKey']
  sessionId: string | null | undefined
}

/**
 * The composer's queue view — the gateway owns the queue (agent.turn_queue)
 * and drains it at the end of every turn, so there is NO auto-drain here and
 * no drain lock. This hook covers what's left client-side: the per-session
 * mirror binding, in-place queued-prompt editing (begin/step/exit), enqueueing
 * the current draft, and "send now" (promote + interrupt on the gateway).
 */
export function useComposerQueue({
  activeQueueSessionKey,
  attachments,
  busy,
  clearDraft,
  draftRef,
  focusInput,
  loadIntoComposer,
  onQueue,
  queueEditRef,
  queueSessionKey,
  sessionId: _sessionId
}: UseComposerQueueArgs) {
  const scope = useComposerScope()

  // Per-session slice (edge): re-renders only when THIS session's queue changes,
  // not on cross-session queue churn (the plain atom's map ref changes on every
  // write; the keyed array does not).
  const queuedPrompts = useSessionSlice($queuedPromptsBySession, activeQueueSessionKey)

  // Steers accepted by the gateway but not yet injected into the live turn.
  const pendingSteers = useSessionSlice($pendingSteersBySession, activeQueueSessionKey)

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

      if (!text.trim() && attachments.length === 0) {
        return false
      }

      const saved = updateQueuedPrompt(queueEdit.sessionKey, queueEdit.entryId, { text })
      triggerHaptic(saved ? 'success' : 'selection')
    } else {
      triggerHaptic('cancel')
    }

    setQueueEditSnapshot(null)
    loadIntoComposer(queueEdit.draft, queueEdit.attachments)
    focusInput()

    return true
  }

  // Queue the current draft on the gateway. onQueue (use-prompt-actions'
  // queuePromptText) resolves attachments into refs and fires
  // session.queue.add; the gateway drains it as the next turn, so nothing
  // more happens client-side. Clears the draft only after the gateway
  // accepts — a rejected enqueue keeps the words in the composer.
  const queueCurrentDraft = useCallback(async () => {
    const text = draftRef.current

    if (!activeQueueSessionKey || !onQueue || (!text.trim() && attachments.length === 0)) {
      return false
    }

    const accepted = await Promise.resolve(onQueue(text, { attachments: cloneAttachments(attachments) }))

    if (!accepted) {
      return false
    }

    clearDraft()
    scope.attachments.clear()
    triggerHaptic('selection')

    return true
  }, [activeQueueSessionKey, attachments, clearDraft, draftRef, onQueue, scope.attachments])

  // "Send now": promote the entry to the queue head and interrupt the live
  // turn on the gateway (queue preserved). The gateway drains the promoted
  // entry the moment the turn unwinds — no client-side send at all. When
  // idle the gateway drains promoted entries on the same RPC.
  const sendQueuedNow = useCallback(
    (id: string) => {
      if (!activeQueueSessionKey || id === queueEdit?.entryId) {
        return false
      }

      triggerHaptic('selection')

      return promoteQueuedPrompt(activeQueueSessionKey, id, { interrupt: busy })
    },
    [activeQueueSessionKey, busy, queueEdit]
  )

  // Manual "fire the next queued turn" gesture (Cmd/Ctrl+Shift+K, empty
  // Enter). The gateway normally drains on its own the moment the session
  // idles; this nudges a head entry that got stuck (e.g. its idle-drain
  // attempt failed while the backend was restarting).
  const drainNextQueued = useCallback(() => {
    const head = queuedPrompts.find(e => e.id !== queueEditRef.current?.entryId)

    return head ? sendQueuedNow(head.id) : false
  }, [queueEditRef, queuedPrompts, sendQueuedNow])

  // Re-key on a runtime session-id change. A stable stored id (queueSessionKey)
  // never churns, so a change there is a real session switch and must NOT
  // migrate; only the runtime-derived key (queueSessionKey falsy → key is
  // sessionId) churns on a backend bounce/resume of the same conversation.
  // Local-mirror-only: the gateway re-syncs authoritative state via
  // session.info/queue.updated after the resume.
  useEffect(() => {
    const prev = prevQueueKeyRef.current
    prevQueueKeyRef.current = activeQueueSessionKey

    if (queueSessionKey || !prev || !activeQueueSessionKey || prev === activeQueueSessionKey) {
      return
    }

    migrateQueuedPrompts(prev, activeQueueSessionKey)
  }, [activeQueueSessionKey, queueSessionKey])

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
    pendingSteers,
    queueCurrentDraft,
    queueEdit,
    queuedPrompts,
    sendQueuedNow,
    stepQueuedEdit
  }
}

import { useStore } from '@nanostores/react'
import { type RefObject, useCallback, useEffect, useRef, useState } from 'react'

import { useI18n } from '@/i18n'
import { isSlashCommandText } from '@/lib/chat-runtime'
import { triggerHaptic } from '@/lib/haptics'
import { useSessionSlice } from '@/lib/use-session-slice'
import { type ComposerAttachment, freezeComposerTransportPayload } from '@/store/composer'
import { resetBrowseState } from '@/store/composer-input-history'
import {
  $parkedQueueSessions,
  $queuedPromptsBySession,
  clearQueuedPromptDrainFailures,
  enqueueQueuedPrompt,
  getQueuedPrompts,
  getQueueParkRevision,
  isSteerableEntry,
  MAX_AUTO_DRAIN_ATTEMPTS,
  migrateQueuedPrompts,
  parkQueuedPrompts,
  type QueuedPromptEntry,
  removeQueuedPrompt,
  resolveQueuedPromptTransport,
  shouldAutoDrain,
  unparkQueuedPrompts,
  updateQueuedPrompt,
  withQueueDrainClaim
} from '@/store/composer-queue'
import { notify } from '@/store/notifications'
import { $sessionsLoading } from '@/store/session'

import { cloneAttachments, type QueueEditState } from '../composer-utils'
import { useComposerScope } from '../scope'
import type { ChatBarProps } from '../types'

/** Freeze terminal chips for queue persistence. Persists the chip/display
 *  form; fenced selection CONTENTS stay in the runtime map. Returns null when
 *  a chip has no selection payload (caller should abort without mutating the queue). */
function freezeQueuedDraftText(
  text: string,
  copy: {
    terminalSelectionMissingTitle: string
    terminalSelectionMissingBody: string
  }
): null | { text: string; displayText?: string; frozenTransport?: string } {
  const trimmed = text.trim()

  if (!trimmed) {
    return { text }
  }

  const frozen = freezeComposerTransportPayload(trimmed)

  if (frozen.missingLabels.length > 0) {
    notify({
      kind: 'warning',
      title: copy.terminalSelectionMissingTitle,
      message: copy.terminalSelectionMissingBody
    })

    return null
  }

  const hasTerminalTransport = frozen.displayText !== frozen.transportText

  return {
    // Persist chips (or ordinary text). Never persist fenced selection contents.
    text: frozen.displayText,
    ...(hasTerminalTransport ? { displayText: frozen.displayText, frozenTransport: frozen.transportText } : {})
  }
}

interface UseComposerQueueArgs {
  activeQueueSessionKey: string | null
  attachments: ComposerAttachment[]
  busy: boolean
  clearDraft: () => void
  draftRef: RefObject<string>
  focusInput: () => void
  loadIntoComposer: (text: string, attachments: ComposerAttachment[]) => void
  onCancel: ChatBarProps['onCancel']
  onSteer: ChatBarProps['onSteer']
  onSubmit: ChatBarProps['onSubmit']
  queueEditRef: RefObject<QueueEditState | null>
  queueSessionKey: ChatBarProps['queueSessionKey']
  sessionId: string | null | undefined
}

/**
 * The composer's queue engine — everything about queued turns: the per-session
 * queue store binding, in-place queued-prompt editing (begin/step/exit), the
 * shared drain lock + send-then-remove sequence, manual send-now, and the
 * edge-independent auto-drain with bounded retries. It consumes the draft API
 * (draftRef/clearDraft/loadIntoComposer/focusInput) and writes the
 * coordinator-owned `queueEditRef` so the draft engine can read the edit state
 * without a back-reference. Manual per-entry sends preserve queue parking.
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
  onSteer,
  onSubmit,
  queueEditRef,
  queueSessionKey,
  sessionId
}: UseComposerQueueArgs) {
  const { t } = useI18n()
  const scope = useComposerScope()

  // Per-session slice (edge): re-renders only when THIS session's queue changes,
  // not on cross-session queue churn (the plain atom's map ref changes on every
  // write; the keyed array does not).
  const queuedPrompts = useSessionSlice($queuedPromptsBySession, activeQueueSessionKey)

  // Parked = the user explicitly halted this session (Stop/Esc) while prompts
  // were queued. The map is tiny (only halted sessions) so a plain subscribe
  // is fine; the auto-drain effect below reads it as a gate.
  const parkedSessions = useStore($parkedQueueSessions)
  const queueParked = Boolean(activeQueueSessionKey && parkedSessions[activeQueueSessionKey])
  const sessionsLoading = useStore($sessionsLoading)

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
  // A per-row Send now is a one-shot request, not a "resume the whole queue" gesture.
  // Capture the session key so a late settle cannot dispatch into another session.
  const pendingManualSendRef = useRef<{
    sessionKey: string
    entryId: string
    parkRevision: number
    cancelAcknowledged: boolean
  } | null>(null)
  const [manualSendRetryTick, setManualSendRetryTick] = useState(0)
  const [drainRetryTick, setDrainRetryTick] = useState(0)

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
    // Edit what the panel SHOWS. A queued `/skill` entry's text is the
    // expanded skill body — never drop that into the composer.
    loadIntoComposer(entry.displayText ?? entry.text, entry.attachments)
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

    const frozen = draftRef.current.trim()
      ? freezeQueuedDraftText(draftRef.current, t.composer)
      : { text: draftRef.current }

    if (!frozen) {
      return true
    }

    const saved = updateQueuedPrompt(queueEdit.sessionKey, queueEdit.entryId, {
      attachments: cloneAttachments(attachments),
      text: frozen.text,
      displayText: frozen.displayText ?? null,
      frozenTransport: frozen.frozenTransport ?? null
    })

    const next = queuedPrompts[target]

    if (next) {
      setQueueEditSnapshot({ ...queueEdit, entryId: next.id })
      loadIntoComposer(next.displayText ?? next.text, next.attachments)
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

      // Editing a queued entry into a slash-command + attachment combo would
      // produce an undrainable entry (submitText rejects it on every attempt).
      // Refuse the save so the queue can never hold an entry that livelocks.
      if (isSlashCommandText(text) && next.length) {
        notify({
          kind: 'warning',
          title: t.desktop.slashCommandIgnoredTitle,
          message: t.desktop.slashCommandIgnoredBody
        })

        return false
      }

      const frozen = text.trim() ? freezeQueuedDraftText(text, t.composer) : { text }

      if (!frozen) {
        return false
      }

      const saved = updateQueuedPrompt(queueEdit.sessionKey, queueEdit.entryId, {
        attachments: next,
        text: frozen.text,
        displayText: frozen.displayText ?? null,
        frozenTransport: frozen.frozenTransport ?? null
      })

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

    // Slash commands cannot ride alongside attachments — the drain path would
    // reject the entry on every attempt (submitText warns + returns false) and
    // the queue would livelock. Refuse to enqueue it in the first place.
    if (isSlashCommandText(text) && attachments.length) {
      notify({
        kind: 'warning',
        title: t.desktop.slashCommandIgnoredTitle,
        message: t.desktop.slashCommandIgnoredBody
      })

      return false
    }

    // Freeze `@terminal:` chips at enqueue. Persist the chip form; keep the
    // fenced selection in the runtime map so drain never re-resolves the live
    // label map and localStorage never stores terminal CONTENTS (#77078).
    const frozen = text.trim() ? freezeQueuedDraftText(text, t.composer) : { text }

    if (!frozen) {
      return false
    }

    if (
      !enqueueQueuedPrompt(activeQueueSessionKey, {
        text: frozen.text,
        attachments,
        ...(frozen.displayText ? { displayText: frozen.displayText } : {}),
        ...(frozen.frozenTransport ? { frozenTransport: frozen.frozenTransport } : {})
      })
    ) {
      return false
    }

    clearDraft()
    // Queue entry retains blob: previews; revoke when the entry is discarded
    // or drained into a submit that takes ownership (see composer-queue).
    scope.attachments.clear({ retainPreviewUrls: true })
    triggerHaptic('selection')

    return true
  }, [activeQueueSessionKey, attachments, clearDraft, draftRef, scope.attachments, t.composer])

  // All queue drain paths share one lock + send-then-remove sequence.
  // `pickEntry` lets each caller choose head, by-id, or skip-edited, from the
  // queue as it stands inside the cross-window claim. Resolves null when there
  // is nothing to send: another window already sent the picked entry.
  const runDrain = useCallback(
    async (
      pickEntry: (entries: QueuedPromptEntry[]) => QueuedPromptEntry | undefined,
      resumeQueue = true
    ): Promise<boolean | null> => {
      if (drainingQueueRef.current || !activeQueueSessionKey) {
        return false
      }

      const drainQueueSessionKey = activeQueueSessionKey
      const drainRuntimeSessionId = sessionId ?? null

      drainingQueueRef.current = true
      const parkRevisionAtStart = getQueueParkRevision(drainQueueSessionKey)

      try {
        return await withQueueDrainClaim(drainQueueSessionKey, async queue => {
          const entry = pickEntry(queue)

          if (!entry) {
            return null
          }

          const resolved = resolveQueuedPromptTransport(entry)

          if (!resolved.ok) {
            notify({
              kind: 'warning',
              title: t.composer.terminalSelectionMissingTitle,
              message: t.composer.queuedTerminalSelectionExpiredBody
            })
            drainFailuresRef.current.set(entry.id, MAX_AUTO_DRAIN_ATTEMPTS)

            return false
          }

          const accepted = await Promise.resolve(
            onSubmit(resolved.transportText, {
              attachments: entry.attachments,
              ...(resolved.displayText ? { displayText: resolved.displayText } : {}),
              ...(entry.displayKind ? { displayKind: entry.displayKind } : {}),
              fromQueue: true,
              sessionId: drainRuntimeSessionId,
              storedSessionId: drainQueueSessionKey
            })
          )

          if (accepted === false) {
            return false
          }

          drainFailuresRef.current.delete(entry.id)
          // Submit now owns the blob: previews (optimistic bubble); do not revoke.
          removeQueuedPrompt(drainQueueSessionKey, entry.id, { retainPreviewUrls: true })
          resetBrowseState(drainRuntimeSessionId)
          // Resuming a queue is different from sending an individual entry.
          // One-shot sends leave the rest parked until the user resumes explicitly.
          // A subsequent Stop / Send now must not be undone by this async submit.
          if (resumeQueue && getQueueParkRevision(drainQueueSessionKey) === parkRevisionAtStart) {
            unparkQueuedPrompts(drainQueueSessionKey)
          }

          return true
        })
      } finally {
        drainingQueueRef.current = false
        // An explicitly selected entry may be waiting on this drain lock.
        if (pendingManualSendRef.current) {
          setManualSendRetryTick(tick => tick + 1)
        }
      }
    },
    [activeQueueSessionKey, onSubmit, sessionId, t.composer]
  )

  const pickDrainHead = useCallback(
    (entries: QueuedPromptEntry[]) => {
      const skip = queueEditRef.current?.entryId

      return skip ? entries.find(e => e.id !== skip) : entries[0]
    },
    [queueEditRef] // reads the edit id off a ref so the lock-holder always sees the latest
  )

  const drainNextQueued = useCallback(async () => (await runDrain(pickDrainHead)) === true, [pickDrainHead, runDrain])

  const sendQueuedNow = useCallback(
    async (id: string): Promise<boolean> => {
      if (
        !activeQueueSessionKey ||
        id === queueEdit?.entryId ||
        !getQueuedPrompts(activeQueueSessionKey).some(entry => entry.id === id)
      ) {
        // A stale row removed in another window must not interrupt a turn.
        return false
      }

      const previous = pendingManualSendRef.current

      if (
        previous &&
        previous.sessionKey === activeQueueSessionKey &&
        getQueueParkRevision(previous.sessionKey) === previous.parkRevision
      ) {
        // Cancellation is already in flight; let the user reselect the row.
        previous.entryId = id
        triggerHaptic('selection')

        return true
      }

      pendingManualSendRef.current = null
      // Also park an initially unparked idle queue. This is a one-shot, and
      // leaving auto-drain enabled would race the selected row.
      parkQueuedPrompts(activeQueueSessionKey)

      const pending = {
        sessionKey: activeQueueSessionKey,
        entryId: id,
        parkRevision: getQueueParkRevision(activeQueueSessionKey),
        cancelAcknowledged: !busy
      }

      if (busy) {
        pendingManualSendRef.current = pending
        triggerHaptic('selection')

        const onCancelFailed = () => {
          if (
            pendingManualSendRef.current !== pending ||
            getQueueParkRevision(pending.sessionKey) !== pending.parkRevision
          ) {
            return
          }

          pendingManualSendRef.current = null
          notify({ kind: 'error', title: t.desktop.stopFailed, message: t.composer.queueStuckTitle })
        }

        try {
          // Ordinary Stop catches and notifies. The strict variant must reject
          // on an RPC failure instead of acknowledging a failed interruption.
          void Promise.resolve(onCancel({ throwOnFailure: true }))
            .then(() => {
              if (pendingManualSendRef.current === pending) {
                pending.cancelAcknowledged = true
                setManualSendRetryTick(tick => tick + 1)
              }
            })
            .catch(onCancelFailed)
        } catch {
          onCancelFailed()

          return false
        }

        return true
      }

      // An older drain may still be settling when the UI first renders idle.
      if (drainingQueueRef.current || sessionsLoading) {
        pendingManualSendRef.current = pending

        return true
      }

      drainFailuresRef.current.delete(id)
      clearQueuedPromptDrainFailures(activeQueueSessionKey, id)

      try {
        return (await runDrain(entries => entries.find(entry => entry.id === id), false)) === true
      } catch {
        notify({ kind: 'error', title: t.composer.queueStuckTitle, message: t.composer.queueStuckBody })

        return false
      }
    },
    [activeQueueSessionKey, busy, onCancel, queueEdit, runDrain, sessionsLoading, t.composer, t.desktop.stopFailed]
  )

  // Deliver a queued entry as a mid-turn redirect — the queue-panel sibling of
  // the composer's steer-on-Enter. No interrupt, no drain lock: a redirect
  // rides the live turn (the gateway either restarts the active request with
  // its displayed context or waits for the current tool boundary), so the turn
  // keeps flowing and the remaining queue is untouched. Only meaningful while
  // busy — idle has no turn to redirect, and `sendQueuedNow` already covers it.
  const steerQueuedNow = useCallback(
    async (id: string): Promise<boolean> => {
      if (!onSteer || !busy || !activeQueueSessionKey || id === queueEditRef.current?.entryId) {
        return false
      }

      const entry = getQueuedPrompts(activeQueueSessionKey).find(e => e.id === id)

      if (!entry || !isSteerableEntry(entry)) {
        return false
      }

      const resolved = resolveQueuedPromptTransport(entry)

      if (!resolved.ok) {
        notify({
          kind: 'warning',
          title: t.composer.terminalSelectionMissingTitle,
          message: t.composer.queuedTerminalSelectionExpiredBody
        })

        return false
      }

      triggerHaptic('submit')

      const accepted = await Promise.resolve(onSteer(resolved.transportText))

      // Rejected (turn already settling, gateway said no): keep this queued
      // entry. Auto-drain picks it up only when the queue is not parked.
      // Only a delivered redirect consumes the entry.
      if (!accepted) {
        return false
      }

      drainFailuresRef.current.delete(id)
      removeQueuedPrompt(activeQueueSessionKey, id)
      // Steering this entry must not silently resume previously parked messages.

      return true
    },
    [activeQueueSessionKey, busy, onSteer, queueEditRef, t.composer]
  )

  // Edge-independent auto-drain: send the head whenever the session is idle and
  // the queue is non-empty, bounding retries so a thrown/rejected onSubmit (e.g.
  // a stale-session 404) can't strand the entry permanently nor spin-loop. The
  // drain lock serializes sends; a remount/reconnect resets the failure counts.
  const autoDrainNext = useCallback(() => {
    if (busy || queueParked || drainingQueueRef.current || !activeQueueSessionKey) {
      return
    }

    const entry = pickDrainHead(queuedPrompts)

    if (!entry || (drainFailuresRef.current.get(entry.id) ?? 0) >= MAX_AUTO_DRAIN_ATTEMPTS) {
      return
    }

    let cancelled = false
    let retryTimer: ReturnType<typeof setTimeout> | undefined

    const onFail = () => {
      if (cancelled) {
        return
      }

      const fails = (drainFailuresRef.current.get(entry.id) ?? 0) + 1
      drainFailuresRef.current.set(entry.id, fails)

      if (fails >= MAX_AUTO_DRAIN_ATTEMPTS) {
        notify({
          id: 'composer-queue-stuck',
          kind: 'error',
          title: t.composer.queueStuckTitle,
          message: t.composer.queueStuckBody
        })
      } else {
        retryTimer = setTimeout(() => setDrainRetryTick(tick => tick + 1), 750 * fails)
      }
    }

    // By id: inside the claim the head may already be gone — sent by another
    // window — which is not a failed send.
    void runDrain(entries => entries.find(e => e.id === entry.id))
      .then(sent => {
        if (sent === false) {
          onFail()
        }
      })
      .catch(onFail)

    // A pending rejection must not schedule into a different session, a parked
    // queue, or an unmounted composer.
    return () => {
      cancelled = true
      clearTimeout(retryTimer)
    }
  }, [activeQueueSessionKey, busy, pickDrainHead, queueParked, queuedPrompts, runDrain, t])

  // Re-key on a runtime session-id change. A stable stored id (queueSessionKey)
  // never churns, so a change there is a real session switch and must NOT
  // migrate; only the runtime-derived key (queueSessionKey falsy → key is
  // sessionId) churns on a backend bounce/resume of the same conversation.
  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    const prev = prevQueueKeyRef.current
    prevQueueKeyRef.current = activeQueueSessionKey

    if (queueSessionKey || !prev || !activeQueueSessionKey || prev === activeQueueSessionKey) {
      return
    }

    migrateQueuedPrompts(prev, activeQueueSessionKey)
  }, [activeQueueSessionKey, queueSessionKey])

  // Queued turns flow whenever the session is idle — on the busy→false settle
  // edge, on mount/reconnect, and after a re-key — so a swallowed edge can't
  // strand them. A park (explicit Stop/Esc) is the one gate: those entries wait
  // for the user. To cancel queued turns, the user deletes them from the panel.
  useEffect(() => {
    // Match the background drainer: preserve the retry budget while session
    // discovery runs at boot, on a gateway/profile switch, or over an empty list.
    if (sessionsLoading) {
      return
    }

    if (shouldAutoDrain({ isBusy: busy, parked: queueParked, queueLength: queuedPrompts.length })) {
      return autoDrainNext()
    }
  }, [autoDrainNext, busy, drainRetryTick, queueParked, queuedPrompts.length, sessionsLoading])

  // A manual one-shot bypasses auto-drain only after cancellation succeeds and
  // the session is idle. Explicit Stop/Esc (including the transcript's Stop)
  // increments the park revision and invalidates the pending send.
  useEffect(() => {
    const pending = pendingManualSendRef.current

    if (!pending) {
      return
    }

    if (
      pending.sessionKey !== activeQueueSessionKey ||
      getQueueParkRevision(pending.sessionKey) !== pending.parkRevision
    ) {
      pendingManualSendRef.current = null
      return
    }

    if (busy || sessionsLoading || !pending.cancelAcknowledged || drainingQueueRef.current) {
      return
    }

    pendingManualSendRef.current = null
    drainFailuresRef.current.delete(pending.entryId)
    clearQueuedPromptDrainFailures(pending.sessionKey, pending.entryId)
    void runDrain(entries => entries.find(entry => entry.id === pending.entryId), false).catch(() => {
      notify({ kind: 'error', title: t.composer.queueStuckTitle, message: t.composer.queueStuckBody })
    })
  }, [activeQueueSessionKey, busy, manualSendRetryTick, runDrain, sessionsLoading, t.composer])

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
    queueCurrentDraft,
    queueEdit,
    queueParked,
    queuedPrompts,
    sendQueuedNow,
    steerQueuedNow,
    stepQueuedEdit
  }
}

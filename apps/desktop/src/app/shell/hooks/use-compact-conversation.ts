import { useStore } from '@nanostores/react'
import { atom } from 'nanostores'
import { useCallback, useEffect, useMemo, useState } from 'react'

import type { CompressSessionOutcome } from '@/app/session/compress-session'
import { translateNow, useI18n } from '@/i18n'
import {
  $compactingSessions,
  markCompressDeferred,
  observeCompactionTerminal,
  setSessionCompacting
} from '@/store/compaction'
import { dismissNotification, notify, notifyError } from '@/store/notifications'

/** Sessions with a manual compaction in flight, keyed by runtime id.
 *
 *  Module-level (not component state) because the action is LLM-bound and
 *  routinely outlives the popover that started it: the dropdown closes, the
 *  panel unmounts, the promise keeps running. Keeping the claim here means a
 *  reopened panel still says "Compacting…" and a second click cannot fire a
 *  second summarise call. Per-session, so navigating to another chat shows
 *  that chat's own state. */
export const $manualCompactingSessions = atom<Record<string, true>>({})

function setManualCompacting(sessionId: string, active: boolean): void {
  const sessions = $manualCompactingSessions.get()
  const current = Boolean(sessions[sessionId])

  if (current === active) {
    return
  }

  const next = { ...sessions }

  if (active) {
    next[sessionId] = true
  } else {
    delete next[sessionId]
  }

  $manualCompactingSessions.set(next)
}

/** Test isolation: the atom is module-level and would otherwise leak a running
 *  claim into the next test. */
export function _resetManualCompactingForTests(): void {
  $manualCompactingSessions.set({})
}

/** Take the running claim for one chat, or report that someone holds it.
 *
 *  Synchronous and store-authoritative on purpose: a double-click on a slow
 *  frame runs the SAME rendered handler twice, so both calls read the same
 *  render snapshot and a snapshot check lets the second summarise through. The
 *  atom is the one value both calls must agree on. */
function claimManualCompacting(claimId: string): boolean {
  if ($manualCompactingSessions.get()[claimId]) {
    return false
  }

  setManualCompacting(claimId, true)

  return true
}

export interface CompactConversationAction {
  /** A turn is running — the composer owns the session, compaction waits. */
  busy: boolean
  confirming: boolean
  error: null | string
  onCancel: () => void
  onConfirm: () => void
  onRequest: () => void
  /** A compaction for THIS chat is already running (this panel or another). */
  pending: boolean
  /** No focused runtime session, or no controller action to call. */
  unavailable: boolean
}

interface CompactConversationOptions {
  /** The controller's owner-routed compress; resolves when the turn is over. */
  compress?: (sessionId: string) => Promise<CompressSessionOutcome>
  busy: boolean
  sessionId: null | string
  /** Durable id of the chat (its stored row), when the caller knows it.
   *
   *  The running claim is keyed on this instead of the runtime id because
   *  stale-runtime recovery re-mints a dead runtime id WHILE the summarise is
   *  still in flight. A claim left on the dead id re-enabled the button over
   *  that call, so a click sent a second summarise for a chat already
   *  compressing. */
  storedSessionId?: null | string
}

/** The context popover's "Compact conversation" action: a two-step
 *  confirmation that captures the chat it was opened on, blocks a duplicate
 *  submit, and reports the outcome where the user can still see it.
 *
 *  The session is CAPTURED at the first click and carried through the confirm
 *  and the RPC. Reading the live focus again at confirm time is how a click on
 *  chat A would compress chat B after a switch — the action belongs to the
 *  conversation whose numbers were on screen. */
export function useCompactConversation({
  busy,
  compress,
  sessionId,
  storedSessionId
}: CompactConversationOptions): CompactConversationAction {
  const { t } = useI18n()
  const copy = t.shell.statusbar.contextUsagePanel.compact
  const compacting = useStore($manualCompactingSessions)
  const backendCompacting = useStore($compactingSessions)
  const [confirmingFor, setConfirmingFor] = useState<null | string>(null)
  // A failure is recorded against the CHAT that failed. The refusal of an
  // LLM-bound compression routinely lands after the user moved on; a bare
  // string painted it under whatever chat was on screen by then.
  const [errorFor, setErrorFor] = useState<null | { claimId: string; message: string }>(null)
  // The durable chat the claim belongs to; empty while neither id is known.
  const durableClaimId = storedSessionId?.trim() || ''
  const claimId = sessionId ? durableClaimId || sessionId : ''
  const error = errorFor && errorFor.claimId === claimId ? errorFor.message : null

  const clearErrorFor = useCallback((target: string) => {
    setErrorFor(current => (current && current.claimId === target ? null : current))
  }, [])

  // A confirmation belongs to the chat it was opened on. Switching chats
  // cancels it (the running claim does NOT cancel — it still belongs to the
  // captured chat and its result lands there).
  useEffect(() => {
    setConfirmingFor(null)
  }, [sessionId])

  useEffect(() => {
    if (busy) {
      setConfirmingFor(null)
    }
  }, [busy])

  const pending = Boolean((claimId && compacting[claimId]) || (sessionId && backendCompacting[sessionId]))

  const onRequest = useCallback(() => {
    if (
      !sessionId ||
      !compress ||
      busy ||
      $compactingSessions.get()[sessionId] ||
      compacting[durableClaimId || sessionId]
    ) {
      return
    }

    clearErrorFor(durableClaimId || sessionId)
    setConfirmingFor(sessionId)
  }, [busy, clearErrorFor, compacting, compress, durableClaimId, sessionId])

  const onCancel = useCallback(() => setConfirmingFor(null), [])

  const onConfirm = useCallback(() => {
    const target = confirmingFor

    // Clear the confirmation first: the claim below is set synchronously, so a
    // second click (a double-click on a slow frame) has nothing left to submit.
    setConfirmingFor(null)

    if (!target || !compress || busy || $compactingSessions.get()[target]) {
      return
    }

    // Held against the CHAT (durable id when the caller knows it), so a
    // stale-runtime re-mint mid-flight cannot orphan it.
    const claimed = durableClaimId || target

    // Claimed before the call, against the store rather than this render's
    // snapshot: a second click in the same frame finds the claim already held.
    if (!claimManualCompacting(claimed)) {
      return
    }

    clearErrorFor(claimed)

    const noticeId = `session-compress:${target}`

    notify({ durationMs: 0, id: noticeId, kind: 'info', message: copy.progress })

    const terminal = observeCompactionTerminal()

    void compress(target)
      .then(outcome => {
        const outcomeNoticeId = `session-compress:${outcome.sessionId}`

        if (outcomeNoticeId !== noticeId) {dismissNotification(noticeId)}

        if (outcome.kind === 'deferred') {
          const ended = terminal.get(outcome.sessionId)

          if (ended) {
            dismissNotification(outcomeNoticeId)

            if (ended.kind === 'compacted') {
              notify({
                durationMs: 5_000,
                id: outcomeNoticeId,
                kind: 'success',
                message: ended.text || translateNow('notifications.compressDeferredDone')
              })
            }

            return
          }

          markCompressDeferred(outcome.sessionId)
          setSessionCompacting(outcome.sessionId, true)

          notify({
            durationMs: 8_000,
            id: outcomeNoticeId,
            kind: 'info',
            message: outcome.message || copy.pending
          })

          return
        }

        // Answered, but nothing was compressed (lock held elsewhere, a no-op,
        // a read-only preview): the backend's own line, not a success claim.
        if (outcome.kind === 'skipped') {
          notify({ durationMs: 8_000, id: outcomeNoticeId, kind: 'info', message: outcome.message })

          return
        }

        notify({
          durationMs: 5_000,
          id: outcomeNoticeId,
          kind: outcome.aborted ? 'error' : 'success',
          message: outcome.message
        })
      })
      .catch((err: unknown) => {
        dismissNotification(noticeId)
        notifyError(err, copy.failed)
        setErrorFor({
          claimId: claimed,
          message: err instanceof Error && err.message ? err.message : copy.failed
        })
      })
      .finally(() => {
        terminal.dispose()
        setManualCompacting(claimed, false)
      })
  }, [busy, clearErrorFor, compress, confirmingFor, copy.failed, copy.pending, copy.progress, durableClaimId])

  return useMemo(
    () => ({
      busy,
      confirming: confirmingFor !== null && confirmingFor === sessionId,
      error,
      onCancel,
      onConfirm,
      onRequest,
      pending,
      unavailable: !sessionId || !compress
    }),
    [busy, confirmingFor, error, onCancel, onConfirm, onRequest, pending, sessionId, compress]
  )
}

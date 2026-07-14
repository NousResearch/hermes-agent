import { type MutableRefObject, useCallback } from 'react'

import { PROMPT_SUBMIT_REQUEST_TIMEOUT_MS } from '@/hermes'
import type { Translations } from '@/i18n'
import { type ChatMessage, textPart } from '@/lib/chat-messages'
import { optimisticAttachmentRef } from '@/lib/chat-runtime'
import { setMutableRef } from '@/lib/mutable-ref'
import {
  $composerAttachments,
  clearComposerAttachments,
  type ComposerAttachment,
  terminalContextBlocksFromDraft
} from '@/store/composer'
import { clearNotifications, notify, notifyError } from '@/store/notifications'
import { requestDesktopOnboarding } from '@/store/onboarding'
import { setAwaitingResponse, setBusy, setMessages } from '@/store/session'

import type { ClientSessionState } from '../../../types'
import { sessionContextDrift } from '../session-context-drift'

import {
  _submitInFlight,
  type GatewayRequest,
  inlineErrorMessage,
  isGatewayTimeoutError,
  isProviderSetupError,
  isSessionBusyError,
  isSessionNotFoundError,
  type SubmitTextOptions,
  withSessionBusyRetry
} from './utils'

interface SubmitPromptDeps {
  activeSessionId: string | null
  activeSessionIdRef: MutableRefObject<string | null>
  busyRef: MutableRefObject<boolean>
  copy: Translations['desktop']
  createBackendSessionForSend: (preview?: string | null) => Promise<string | null>
  getRouteToken: () => string
  requestGateway: GatewayRequest
  selectedStoredSessionIdRef: MutableRefObject<string | null>
  syncAttachmentsForSubmit: (
    sessionId: string,
    attachments: ComposerAttachment[],
    options?: { updateComposerAttachments?: boolean }
  ) => Promise<ComposerAttachment[]>
  updateSessionState: (
    sessionId: string,
    updater: (state: ClientSessionState) => ClientSessionState,
    storedSessionId?: string | null
  ) => ClientSessionState
}

/** The prompt submit pipeline, extracted from usePromptActions. */
export function useSubmitPrompt(deps: SubmitPromptDeps) {
  const {
    activeSessionId,
    activeSessionIdRef,
    busyRef,
    copy,
    createBackendSessionForSend,
    getRouteToken,
    requestGateway,
    selectedStoredSessionIdRef,
    syncAttachmentsForSubmit,
    updateSessionState
  } = deps

  return useCallback(
    async (rawText: string, options?: SubmitTextOptions) => {
      const visibleText = rawText.trim()
      const usingComposerAttachments = !options?.attachments

      // Drop undefined/null holes a session switch or draft restore can leave in
      // the attachments array (same bug class as AttachmentList #49624). Without
      // this, the sibling iterations below (a.kind / a.label / a.refText, and the
      // sync step) throw "Cannot read properties of undefined (reading 'refText')"
      // and break the chat surface.
      const attachments = (options?.attachments ?? $composerAttachments.get()).filter((a): a is ComposerAttachment =>
        Boolean(a)
      )

      const terminalContextBlocks = terminalContextBlocksFromDraft(rawText).join('\n\n')
      const hasImage = attachments.some(a => a.kind === 'image')

      // Refs are recomputed after sync (file.attach rewrites @file: refs to
      // workspace-relative paths the remote gateway can resolve). Seed the
      // optimistic message with the pre-sync refs, then rewrite once synced.
      // Images use their base64 preview so the thumbnail renders inline without
      // a (remote-mode 403-prone) /api/media fetch — see optimisticAttachmentRef.
      let attachmentRefs = attachments.map(optimisticAttachmentRef).filter((r): r is string => Boolean(r))

      const buildContextText = (atts: ComposerAttachment[]): string => {
        // atts may be the post-sync array, which can reintroduce holes; filter
        // before touching a.refText / a.kind.
        const present = atts.filter((a): a is ComposerAttachment => Boolean(a))

        const contextRefs = present
          .map(a => a.refText)
          .filter(Boolean)
          .join('\n')

        return (
          [contextRefs, terminalContextBlocks, visibleText].filter(Boolean).join('\n\n') ||
          (present.some(a => a.kind === 'image') ? 'What do you see in this image?' : '')
        )
      }

      // Queue drains fire on the busy→false settle edge, where busyRef (synced
      // from $busy by a separate effect) may still read true — honoring it would
      // bounce the drained send. The drain lock serializes them; the user path
      // keeps the guard so a stray Enter mid-turn can't double-submit.
      const hasSendable = Boolean(visibleText || terminalContextBlocks || attachments.length || hasImage)

      if (!hasSendable || (!options?.fromQueue && busyRef.current)) {
        return false
      }

      // Pin the session context for the whole async submit pipeline. Without
      // this, a fast session switch during session.resume / file.attach can
      // redirect the user's text into a different chat (#54527). Mutable —
      // not const — because a new-chat submit legitimately re-homes to the
      // session it creates (see the re-pin after createBackendSessionForSend).
      const startingActiveSessionId = activeSessionIdRef.current
      let startingStoredSessionId = selectedStoredSessionIdRef.current
      let startingRouteToken = getRouteToken()

      // Reason string (or null) for why the session context genuinely drifted
      // under this in-flight submit. sessionContextDrift ignores the churn a
      // busy gateway produces (selection null-resets on a gateway/profile
      // switch, search/hash-only route changes, background active-ref
      // retargets) so a second-session send doesn't silently abort — it fires
      // only on a real move to a DIFFERENT chat. Reads the live refs/route each
      // call and measures against the (mutable) baseline, which is re-pinned to
      // the created chat after createBackendSessionForSend. submitTargetStoredId
      // is the stored session this submit targets, so a move ONTO it (the
      // pipeline's own re-home) is never counted as drift.
      const sessionDriftReason = (): string | null =>
        sessionContextDrift({
          startRouteToken: startingRouteToken,
          nowRouteToken: getRouteToken(),
          startSelectedStoredId: startingStoredSessionId,
          nowSelectedStoredId: selectedStoredSessionIdRef.current,
          submitTargetStoredId: startingStoredSessionId
        })

      // One submit in flight per session — drop any concurrent re-fire so a
      // stalled turn can't stack the same prompt into multiple real turns.
      const submitLockKey = startingStoredSessionId || startingActiveSessionId || '__pending_new__'

      if (_submitInFlight.has(submitLockKey)) {
        return false
      }

      _submitInFlight.add(submitLockKey)
      let submitLockReleased = false

      const releaseSubmitLock = () => {
        if (!submitLockReleased) {
          submitLockReleased = true
          _submitInFlight.delete(submitLockKey)
        }
      }

      const optimisticId = `user-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`

      const buildUserMessage = (): ChatMessage => ({
        id: optimisticId,
        role: 'user',
        parts: [textPart(visibleText || (attachmentRefs.length ? '' : attachments.map(a => a.label).join(', ')))],
        attachmentRefs
      })

      const releaseBusy = () => {
        releaseSubmitLock()
        setMutableRef(busyRef, false)
        setBusy(false)
        setAwaitingResponse(false)
      }

      // Idempotent optimistic insert — re-running with the resolved sessionId
      // after createBackendSessionForSend just overwrites with the same id.
      const seedOptimistic = (sid: string) =>
        updateSessionState(
          sid,
          state => ({
            ...state,
            messages: state.messages.some(m => m.id === optimisticId)
              ? state.messages
              : [...state.messages, buildUserMessage()],
            busy: true,
            awaitingResponse: true,
            pendingBranchGroup: null,
            sawAssistantPayload: false,
            // Fresh submit = new turn — clear any leftover interrupt flag, else
            // mutateStream/completeAssistantMessage drop every delta of this turn
            // (what made drained-after-interrupt sends go silent).
            interrupted: false
          }),
          startingStoredSessionId
        )

      // After sync rewrites refs, refresh the optimistic message in place so the
      // transcript shows the resolved @file: ref rather than the local path.
      const rewriteOptimistic = (sid: string) =>
        updateSessionState(
          sid,
          state => ({
            ...state,
            messages: state.messages.map(message => (message.id === optimisticId ? buildUserMessage() : message))
          }),
          startingStoredSessionId
        )

      const dropOptimistic = (sid: null | string) => {
        if (!sid) {
          setMessages(current => current.filter(m => m.id !== optimisticId))

          return
        }

        updateSessionState(
          sid,
          state => ({
            ...state,
            messages: state.messages.filter(m => m.id !== optimisticId),
            busy: false,
            awaitingResponse: false,
            pendingBranchGroup: null
          }),
          startingStoredSessionId
        )
      }

      const abortForSessionSwitch = (optimisticSessionId: null | string): false => {
        dropOptimistic(optimisticSessionId)
        releaseBusy()

        return false
      }

      setMutableRef(busyRef, true)
      setBusy(true)
      setAwaitingResponse(true)
      clearNotifications()

      let sessionId: null | string = activeSessionId

      if (sessionId) {
        seedOptimistic(sessionId)
      } else {
        setMessages(current => [...current, buildUserMessage()])
      }

      if (!sessionId && startingStoredSessionId) {
        // A stored session is SELECTED but its runtime binding is gone (the
        // live session was orphan-reaped, or a timeout/reconnect cleared
        // activeSessionId). Continuing the selected conversation must mean
        // resuming it — minting a brand-new backend session here silently
        // splits the user's chat in two (#55578 symptom b). Only fall through
        // to session creation when NO stored session is selected (a genuine
        // new-chat draft).
        try {
          const resumed = await requestGateway<{ session_id: string }>('session.resume', {
            session_id: startingStoredSessionId
          })

          const resumeDrift = sessionDriftReason()

          if (resumeDrift) {
            console.warn('[submit-drift-abort]', resumeDrift, { phase: 'post-resume' })

            return abortForSessionSwitch(sessionId)
          }

          if (resumed?.session_id) {
            sessionId = resumed.session_id
            activeSessionIdRef.current = sessionId
          }
        } catch {
          // Resume failed (session gone from state.db, gateway hiccup) —
          // fall through to creating a fresh session rather than dead-ending
          // the user's message.
        }

        const resumeSettleDrift = sessionDriftReason()

        if (resumeSettleDrift) {
          console.warn('[submit-drift-abort]', resumeSettleDrift, { phase: 'post-resume-settle' })

          return abortForSessionSwitch(sessionId)
        }

        if (sessionId) {
          seedOptimistic(sessionId)
        }
      }

      if (!sessionId) {
        try {
          sessionId = await createBackendSessionForSend(visibleText)
        } catch (err) {
          dropOptimistic(null)
          releaseBusy()
          notifyError(err, copy.sessionUnavailable)

          return false
        }

        if (!sessionId) {
          // createBackendSessionForSend returns null when the user switched
          // sessions mid-create (it closes the orphaned session itself) —
          // abort silently. Anything else is a real failure worth a toast.
          const createNullDrift = sessionDriftReason()

          if (createNullDrift) {
            console.warn('[submit-drift-abort]', createNullDrift, { phase: 'post-create-null' })

            return abortForSessionSwitch(null)
          }

          dropOptimistic(null)
          releaseBusy()
          notify({ kind: 'error', title: copy.sessionUnavailable, message: copy.createSessionFailed })

          return false
        }

        // A successful create re-homes selection + route to the chat it just
        // minted, so the pre-create baseline can't tell our own re-home from
        // a user switch (judging it drift aborted EVERY first send of a new
        // chat: no prompt.submit, no DB row, a stranded route that 404s
        // "Session not found"). The drift signal for this window is the
        // active ref instead: every switch path re-nulls or retargets it
        // synchronously, so it only still equals the id create returned when
        // nobody re-homed since.
        if (activeSessionIdRef.current !== sessionId) {
          return abortForSessionSwitch(sessionId)
        }

        // Re-pin the baseline to the created chat for the rest of the
        // pipeline; the closures (seedOptimistic et al) see the new value.
        startingStoredSessionId = selectedStoredSessionIdRef.current
        startingRouteToken = getRouteToken()

        seedOptimistic(sessionId)
      }

      try {
        const syncedAttachments = await syncAttachmentsForSubmit(sessionId, attachments, {
          updateComposerAttachments: usingComposerAttachments
        })

        const attachmentsDrift = sessionDriftReason()

        if (attachmentsDrift) {
          console.warn('[submit-drift-abort]', attachmentsDrift, { phase: 'post-attachments' })

          return abortForSessionSwitch(sessionId)
        }

        // Rewrite the optimistic message + prompt text with the synced refs so
        // the gateway receives @file: paths that resolve in its workspace.
        // (Images keep their inline base64 preview — see optimisticAttachmentRef.)
        attachmentRefs = syncedAttachments.map(optimisticAttachmentRef).filter((r): r is string => Boolean(r))
        rewriteOptimistic(sessionId)
        const text = buildContextText(syncedAttachments)

        // On sleep/wake the gateway's in-memory session may have been cleared
        // while the desktop app still holds the old session ID. Detect this,
        // resume the stored session to re-register it, and retry once.
        let submitErr: unknown = null

        try {
          await withSessionBusyRetry(() =>
            requestGateway('prompt.submit', { session_id: sessionId, text }, PROMPT_SUBMIT_REQUEST_TIMEOUT_MS)
          )
        } catch (firstErr) {
          if (
            (isSessionNotFoundError(firstErr) || isGatewayTimeoutError(firstErr)) &&
            startingStoredSessionId
          ) {
            // Re-register the session in the gateway and get a fresh live ID.
            // Timeouts recover the same way as "session not found": a starved
            // backend loop (#55578 symptom d) rejects the submit even though
            // the stored session is fine — resume + retry instead of erroring
            // out and losing the session binding.
            const resumed = await requestGateway<{ session_id: string }>('session.resume', {
              session_id: startingStoredSessionId,
              source: 'desktop'
            })

            const resumeRetryDrift = sessionDriftReason()

            if (resumeRetryDrift) {
              console.warn('[submit-drift-abort]', resumeRetryDrift, { phase: 'post-resume-retry' })

              return abortForSessionSwitch(sessionId)
            }

            const recoveredId = resumed?.session_id

            if (recoveredId) {
              activeSessionIdRef.current = recoveredId
              await withSessionBusyRetry(() =>
                requestGateway('prompt.submit', { session_id: recoveredId, text }, PROMPT_SUBMIT_REQUEST_TIMEOUT_MS)
              )
            } else {
              submitErr = firstErr
            }
          } else {
            submitErr = firstErr
          }
        }

        if (submitErr !== null) {
          throw submitErr
        }

        if (usingComposerAttachments) {
          clearComposerAttachments()
        }

        // Submit landed — the turn now runs (busy stays true), but the submit
        // window is closed, so release the lock for the next (sequential) send.
        releaseSubmitLock()

        return true
      } catch (err) {
        releaseBusy()

        // A queued drain that raced a not-yet-settled turn gets a transient
        // "session busy" (4009). Don't surface an error bubble/toast — the entry
        // stays queued and the composer's bounded auto-drain retries when idle.
        if (options?.fromQueue && isSessionBusyError(err)) {
          return false
        }

        const message = inlineErrorMessage(err, copy.promptFailed)

        updateSessionState(sessionId, state => ({
          ...state,
          messages: [
            ...state.messages,
            {
              id: `assistant-error-${Date.now()}`,
              role: 'assistant',
              parts: [],
              error: message || copy.promptFailed,
              branchGroupId: state.pendingBranchGroup ?? undefined
            }
          ],
          busy: false,
          awaitingResponse: false,
          pendingBranchGroup: null,
          sawAssistantPayload: true
        }))

        if (isProviderSetupError(err)) {
          requestDesktopOnboarding(copy.providerCredentialRequired)

          return false
        }

        notifyError(err, copy.promptFailed)

        return false
      }
    },
    [
      activeSessionId,
      activeSessionIdRef,
      busyRef,
      copy,
      createBackendSessionForSend,
      getRouteToken,
      requestGateway,
      selectedStoredSessionIdRef,
      syncAttachmentsForSubmit,
      updateSessionState
    ]
  )
}

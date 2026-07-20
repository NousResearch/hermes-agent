import { type MutableRefObject, useCallback } from 'react'

import { PROMPT_SUBMIT_REQUEST_TIMEOUT_MS } from '@/hermes'
import type { Translations } from '@/i18n'
import { type ChatMessage, textPart } from '@/lib/chat-messages'
import { optimisticAttachmentRef } from '@/lib/chat-runtime'
import { sanitizeComposerInput } from '@/lib/composer-input-sanitize'
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
  activeSessionIdRef: MutableRefObject<string | null>
  busyRef: MutableRefObject<boolean>
  copy: Translations['desktop']
  createBackendSessionForSend: (preview?: string | null) => Promise<string | null>
  getRoutedStoredSessionId: () => null | string
  getRuntimeIdForStoredSession: (storedSessionId: string) => null | string
  getRouteToken: () => string
  requestGateway: GatewayRequest
  resumeStoredSession: (storedSessionId: string) => Promise<void> | void
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
  /** Composer-scope seams: the main chat runs on the module-level globals
   *  (defaults); a session tile injects its own so a tile submit never writes
   *  the primary view's $busy/$messages or clears the main attachment chips. */
  scope?: {
    clearAttachments: () => void
    readAttachments: () => ComposerAttachment[]
    setAwaitingResponse: (awaiting: boolean) => void
    setBusy: (busy: boolean) => void
    setMessages: (updater: (current: ChatMessage[]) => ChatMessage[]) => void
  }
}

// Stable identity — a fresh default object per render would churn the
// useCallback below on every render.
const MAIN_SUBMIT_SCOPE: NonNullable<SubmitPromptDeps['scope']> = {
  clearAttachments: clearComposerAttachments,
  readAttachments: () => $composerAttachments.get(),
  setAwaitingResponse,
  setBusy,
  setMessages
}

/** The prompt submit pipeline, extracted from usePromptActions. */
export function useSubmitPrompt(deps: SubmitPromptDeps) {
  const {
    activeSessionIdRef,
    busyRef,
    copy,
    createBackendSessionForSend,
    getRoutedStoredSessionId,
    getRuntimeIdForStoredSession,
    getRouteToken,
    requestGateway,
    resumeStoredSession,
    selectedStoredSessionIdRef,
    syncAttachmentsForSubmit,
    updateSessionState,
    scope = MAIN_SUBMIT_SCOPE
  } = deps

  return useCallback(
    async (rawText: string, options?: SubmitTextOptions) => {
      const visibleText = sanitizeComposerInput(rawText).trim()
      const usingComposerAttachments = !options?.attachments

      // Drop undefined/null holes a session switch or draft restore can leave in
      // the attachments array (same bug class as AttachmentList #49624). Without
      // this, the sibling iterations below (a.kind / a.label / a.refText, and the
      // sync step) throw "Cannot read properties of undefined (reading 'refText')"
      // and break the chat surface.
      const attachments = (options?.attachments ?? scope.readAttachments()).filter((a): a is ComposerAttachment =>
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

      // Queue drains carry their source session explicitly. A background drain
      // must never inherit the currently selected session after the user moves
      // to another chat.
      const targetStoredSessionId = options?.storedSessionId ?? selectedStoredSessionIdRef.current

      const targetStartedInCurrentView =
        !targetStoredSessionId || targetStoredSessionId === selectedStoredSessionIdRef.current

      let sessionId: null | string = options?.sessionId ?? activeSessionIdRef.current

      // Pin the foreground session context for the whole async submit pipeline.
      // Without this, a fast session switch during session.resume / file.attach
      // can redirect the user's text into a different chat (#54527). Mutable —
      // not const — because a new-chat submit legitimately re-homes to the
      // session it creates (see the re-pin after createBackendSessionForSend).
      const startingActiveSessionId = activeSessionIdRef.current
      const selectedStoredSessionId = selectedStoredSessionIdRef.current
      const routedStoredSessionId = getRoutedStoredSessionId()

      const routedRuntimeId = routedStoredSessionId ? getRuntimeIdForStoredSession(routedStoredSessionId) : null

      const routedSessionNeedsResume = Boolean(
        routedStoredSessionId &&
        (selectedStoredSessionId !== routedStoredSessionId ||
          !startingActiveSessionId ||
          startingActiveSessionId !== routedRuntimeId)
      )

      let startingStoredSessionId = routedSessionNeedsResume
        ? routedStoredSessionId
        : (selectedStoredSessionId ?? routedStoredSessionId)

      let startingRouteToken = getRouteToken()

      // Stored id the optimistic message (and error bubble) is keyed under.
      // Tracks targetStoredSessionId until the dead-draft recovery in the
      // prompt.submit catch below re-homes the send to a freshly minted chat —
      // keying the new runtime under the dead stored id would cross-wire the
      // session-state cache.
      let optimisticStoredSessionId = targetStoredSessionId

      const sessionContextDrifted = (): boolean =>
        targetStartedInCurrentView &&
        (selectedStoredSessionIdRef.current !== startingStoredSessionId || getRouteToken() !== startingRouteToken)

      const targetIsCurrentView = (): boolean => targetStartedInCurrentView && !sessionContextDrifted()

      // One submit in flight per session — drop any concurrent re-fire so a
      // stalled turn can't stack the same prompt into multiple real turns. The
      // foreground ChatBar and background drainers can briefly overlap during a
      // session switch; this per-session lock makes that safe.
      const submitLockKey = targetStoredSessionId || sessionId || startingActiveSessionId || '__pending_new__'

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

        if (targetIsCurrentView()) {
          setMutableRef(busyRef, false)
          scope.setBusy(false)
          scope.setAwaitingResponse(false)
        }
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
          optimisticStoredSessionId
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
          optimisticStoredSessionId
        )

      const dropOptimistic = (sid: null | string) => {
        if (!sid) {
          if (targetIsCurrentView()) {
            scope.setMessages(current => current.filter(m => m.id !== optimisticId))
          }

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
          optimisticStoredSessionId
        )
      }

      const abortForSessionSwitch = (optimisticSessionId: null | string): false => {
        dropOptimistic(optimisticSessionId)
        releaseBusy()

        return false
      }

      // Shared dead-runtime recovery for the two session-scoped failure points
      // in the send pipeline (the initial attachment sync and prompt.submit
      // itself): re-register the stored session in the gateway for a fresh
      // live id. Timeouts recover the same way as "session not found": a
      // starved backend loop (#55578 symptom d) rejects the call even though
      // the stored session is fine — resume instead of erroring out and losing
      // the session binding.
      //
      // The resume can itself fail: the gateway persists a chat's DB row only
      // on its first successful prompt.submit, so a fresh chat whose live
      // session died before that has no row and the resume 4007s the same
      // "session not found". When BOTH the live id and the stored row report
      // "session not found" (the never-persisted-draft signature), mint a
      // replacement chat and re-home the optimistic message there. Scoped
      // tightly: a timed-out call may have actually reached the backend
      // (re-sending elsewhere would double-send), a non-404 resume failure
      // (transport blip) must not split a real stored chat in two (#55578
      // symptom b), and a background drain must never re-home the user's view —
      // all of those return null so the caller surfaces the original error.
      //
      // Returns 'aborted' after a drift abort (the caller must return false),
      // null when the caller should surface the original error, or the live
      // runtime id to continue the send on.
      const recoverDeadRuntime = async (
        originalErr: unknown
      ): Promise<'aborted' | null | { minted: boolean; sessionId: string }> => {
        const recoverStoredSessionId = targetStoredSessionId ?? selectedStoredSessionIdRef.current

        if (!(isSessionNotFoundError(originalErr) || isGatewayTimeoutError(originalErr)) || !recoverStoredSessionId) {
          return null
        }

        let resumed: { session_id: string } | null = null
        let resumeErr: unknown = null

        try {
          resumed = await requestGateway<{ session_id: string }>('session.resume', {
            session_id: recoverStoredSessionId,
            source: 'desktop'
          })
        } catch (err) {
          resumeErr = err
        }

        if (sessionContextDrifted()) {
          abortForSessionSwitch(sessionId)

          return 'aborted'
        }

        const recoveredId = resumed?.session_id

        if (recoveredId) {
          if (targetIsCurrentView()) {
            activeSessionIdRef.current = recoveredId
          }

          return { minted: false, sessionId: recoveredId }
        }

        if (
          !targetIsCurrentView() ||
          !isSessionNotFoundError(originalErr) ||
          resumeErr === null ||
          !isSessionNotFoundError(resumeErr)
        ) {
          return null
        }

        let createdId: null | string = null

        try {
          createdId = await createBackendSessionForSend(visibleText)
        } catch {
          createdId = null
        }

        if (!createdId) {
          // Null means the user switched sessions mid-create (it closes the
          // orphan itself) — abort silently; a create failure keeps the
          // original error.
          if (sessionContextDrifted()) {
            abortForSessionSwitch(sessionId)

            return 'aborted'
          }

          return null
        }

        if (activeSessionIdRef.current !== createdId) {
          // Same re-home race as the primary create path below: every switch
          // path re-nulls or retargets the active ref synchronously, so a
          // mismatch means the user moved on.
          abortForSessionSwitch(sessionId)

          return 'aborted'
        }

        // Move the optimistic message from the dead chat into the one the
        // create re-homed to, and re-pin the drift baseline there.
        dropOptimistic(sessionId)
        optimisticStoredSessionId = selectedStoredSessionIdRef.current
        startingStoredSessionId = selectedStoredSessionIdRef.current
        startingRouteToken = getRouteToken()
        sessionId = createdId
        seedOptimistic(createdId)

        return { minted: true, sessionId: createdId }
      }

      // Foreground-only state: a background queue drain must never write the
      // selected view's busy/awaiting flags or clear its notifications.
      if (targetIsCurrentView()) {
        setMutableRef(busyRef, true)
        scope.setBusy(true)
        scope.setAwaitingResponse(true)
        clearNotifications()
      }

      // A route whose selected/runtime binding is incomplete or cross-wired
      // outranks a stale render-time runtime id (often from the previous
      // profile): force the full routed resume path below. An explicit queued
      // runtime id (background drain) is authoritative and is left untouched.
      if (!options?.sessionId && routedSessionNeedsResume) {
        sessionId = null
      }

      if (sessionId) {
        seedOptimistic(sessionId)
      } else if (targetIsCurrentView()) {
        scope.setMessages(current => [...current, buildUserMessage()])
      }

      if (!sessionId && routedStoredSessionId && routedSessionNeedsResume) {
        // The URL still names a durable conversation, but a profile
        // swap/reconnect left its volatile session binding incomplete or
        // cross-wired. Run the full profile-aware resume path. Creating here
        // would fork a contextless chat against whichever profile is active.
        try {
          await resumeStoredSession(routedStoredSessionId)
        } catch {
          return abortForSessionSwitch(null)
        }

        if (sessionContextDrifted()) {
          return abortForSessionSwitch(null)
        }

        const recoveredRuntimeId = activeSessionIdRef.current
        const validatedRuntimeId = getRuntimeIdForStoredSession(routedStoredSessionId)

        // Recovery only succeeded when both sides of the cache agree that the
        // live runtime belongs to the durable routed session. A failed profile
        // swap may leave the previous profile's runtime active, while a recycled
        // runtime id may leave a cross-wired stored-session mapping.
        if (
          !recoveredRuntimeId ||
          recoveredRuntimeId !== validatedRuntimeId ||
          selectedStoredSessionIdRef.current !== routedStoredSessionId
        ) {
          return abortForSessionSwitch(null)
        }

        sessionId = recoveredRuntimeId
        seedOptimistic(sessionId)
      }

      if (!sessionId && targetStoredSessionId) {
        // A target stored session exists but its runtime binding is gone (the
        // live session was orphan-reaped, a timeout/reconnect cleared it, or a
        // background queue drain only has the durable id). Continue that target
        // conversation; only a genuine new-chat draft may create a new session.
        try {
          const resumed = await requestGateway<{ session_id: string }>('session.resume', {
            session_id: targetStoredSessionId,
            source: 'desktop'
          })

          if (sessionContextDrifted()) {
            return abortForSessionSwitch(sessionId)
          }

          if (resumed?.session_id) {
            sessionId = resumed.session_id

            if (targetIsCurrentView()) {
              activeSessionIdRef.current = sessionId
            }
          }
        } catch {
          // A target stored conversation is not a new-chat draft. If its
          // runtime cannot be rebound, stop here rather than silently replacing
          // it with a contextless session (#55578). For a background/queued
          // drain this abort is a no-op on foreground state (both helpers are
          // targetIsCurrentView-guarded) and simply drops the queued send.
          return abortForSessionSwitch(null)
        }

        if (sessionContextDrifted()) {
          return abortForSessionSwitch(sessionId)
        }

        if (!sessionId) {
          return abortForSessionSwitch(null)
        }

        seedOptimistic(sessionId)
      }

      if (!sessionId) {
        try {
          sessionId = await createBackendSessionForSend(visibleText)
        } catch (err) {
          dropOptimistic(null)
          releaseBusy()

          if (targetIsCurrentView()) {
            notifyError(err, copy.sessionUnavailable)
          }

          return false
        }

        if (!sessionId) {
          // createBackendSessionForSend returns null when the user switched
          // sessions mid-create (it closes the orphaned session itself) —
          // abort silently. Anything else is a real failure worth a toast.
          if (sessionContextDrifted()) {
            return abortForSessionSwitch(null)
          }

          dropOptimistic(null)
          releaseBusy()

          if (targetIsCurrentView()) {
            notify({ kind: 'error', title: copy.sessionUnavailable, message: copy.createSessionFailed })
          }

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
        // file.attach / image.attach are session-scoped RPCs, so a dead
        // runtime fails HERE — before prompt.submit ever runs — whenever the
        // draft has newly selected attachments. Recover through the same
        // resume-or-mint path, then stage the attachments against the live id
        // it produced.
        let syncedAttachments: ComposerAttachment[]

        try {
          syncedAttachments = await syncAttachmentsForSubmit(sessionId, attachments, {
            updateComposerAttachments: usingComposerAttachments
          })
        } catch (syncErr) {
          const recovery = isSessionNotFoundError(syncErr) ? await recoverDeadRuntime(syncErr) : null

          if (recovery === 'aborted') {
            return false
          }

          if (recovery === null) {
            throw syncErr
          }

          if (!recovery.minted) {
            // The resumed runtime replaces the dead one for the rest of the
            // pipeline (a mint already re-homed inside recoverDeadRuntime).
            dropOptimistic(sessionId)
            sessionId = recovery.sessionId
            seedOptimistic(sessionId)
          }

          syncedAttachments = await syncAttachmentsForSubmit(recovery.sessionId, attachments, {
            updateComposerAttachments: usingComposerAttachments
          })
        }

        if (sessionContextDrifted()) {
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
          const recovery = await recoverDeadRuntime(firstErr)

          if (recovery === 'aborted') {
            return false
          }

          if (recovery === null) {
            submitErr = firstErr
          } else {
            const retryId = recovery.sessionId
            let retryText = text

            if (recovery.minted) {
              // Attachments were staged into the dead session — re-sync them
              // against the minted one so @file:/image refs resolve there.
              const resyncedAttachments = await syncAttachmentsForSubmit(retryId, syncedAttachments, {
                updateComposerAttachments: usingComposerAttachments
              })

              if (sessionContextDrifted()) {
                return abortForSessionSwitch(retryId)
              }

              attachmentRefs = resyncedAttachments.map(optimisticAttachmentRef).filter((r): r is string => Boolean(r))
              rewriteOptimistic(retryId)
              retryText = buildContextText(resyncedAttachments)
            }

            await withSessionBusyRetry(() =>
              requestGateway('prompt.submit', { session_id: retryId, text: retryText }, PROMPT_SUBMIT_REQUEST_TIMEOUT_MS)
            )
          }
        }

        if (submitErr !== null) {
          throw submitErr
        }

        if (usingComposerAttachments) {
          scope.clearAttachments()
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

        updateSessionState(
          sessionId,
          state => ({
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
          }),
          optimisticStoredSessionId
        )

        if (targetIsCurrentView() && isProviderSetupError(err)) {
          requestDesktopOnboarding(copy.providerCredentialRequired)

          return false
        }

        if (targetIsCurrentView()) {
          notifyError(err, copy.promptFailed)
        }

        return false
      }
    },
    [
      activeSessionIdRef,
      busyRef,
      copy,
      createBackendSessionForSend,
      getRoutedStoredSessionId,
      getRuntimeIdForStoredSession,
      getRouteToken,
      requestGateway,
      resumeStoredSession,
      scope,
      selectedStoredSessionIdRef,
      syncAttachmentsForSubmit,
      updateSessionState
    ]
  )
}

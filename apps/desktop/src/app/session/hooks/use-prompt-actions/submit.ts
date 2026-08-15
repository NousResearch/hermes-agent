import { type MutableRefObject, useCallback } from 'react'

import { PROMPT_SUBMIT_REQUEST_TIMEOUT_MS } from '@/hermes'
import type { Translations } from '@/i18n'
import { type ChatMessage, textPart } from '@/lib/chat-messages'
import { optimisticAttachmentRef } from '@/lib/chat-runtime'
import { sanitizeComposerInput } from '@/lib/composer-input-sanitize'
import { setMutableRef } from '@/lib/mutable-ref'
import {
  isVoicePlaybackActive,
  markVoicePlaybackInterrupted,
  stopVoicePlayback,
  takeVoicePlaybackInterrupted
} from '@/lib/voice-playback'
import {
  $composerAttachments,
  type ComposerAttachment,
  mainComposerScope,
  terminalContextBlocksFromDraft
} from '@/store/composer'
import { $hudMode } from '@/store/hud'
import { clearNotifications, notify, notifyError } from '@/store/notifications'
import { requestDesktopOnboarding } from '@/store/onboarding'
import {
  $sessions,
  resolveComposerSessionKey,
  setActiveSessionId,
  setAwaitingResponse,
  setBusy,
  setMessages,
  touchSessionActivity
} from '@/store/session'
import { $sessionStates } from '@/store/session-states'

import type { ClientSessionState } from '../../../types'
import { sessionContextDrift } from '../session-context-drift'
import { resolveSessionProfile } from '../use-session-actions/utils'

import { finalizeInterruptedMessages } from './rewind'
import {
  acquireSubmitInFlight,
  type GatewayRequest,
  inlineErrorMessage,
  isProviderSetupError,
  isSessionBusyError,
  isTargetSessionBusy,
  releaseSubmitInFlight,
  SessionRecoveryAborted,
  type SubmitTextOptions,
  withSessionBusyRetry,
  withSessionNotFoundResume
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
  runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>>
  resumeStoredSession: (storedSessionId: string) => Promise<void> | void
  selectedStoredSessionIdRef: MutableRefObject<string | null>
  syncAttachmentsForSubmit: (
    sessionId: string,
    attachments: ComposerAttachment[],
    options?: { updateComposerAttachments?: boolean }
  ) => Promise<{ attachments: ComposerAttachment[]; sessionId: string }>
  updateSessionState: (
    sessionId: string,
    updater: (state: ClientSessionState) => ClientSessionState,
    storedSessionId?: string | null
  ) => ClientSessionState
  /** Composer-scope seams: the main chat runs on the module-level globals
   *  (defaults); a session tile injects its own so a tile submit never writes
   *  the primary view's $busy/$messages or clears the main attachment chips. */
  scope?: {
    removeAttachments: (attachments: readonly ComposerAttachment[]) => void
    readAttachments: () => ComposerAttachment[]
    setAwaitingResponse: (awaiting: boolean) => void
    setBusy: (busy: boolean) => void
    setMessages: (updater: (current: ChatMessage[]) => ChatMessage[]) => void
  }
}

// Stable identity — a fresh default object per render would churn the
// useCallback below on every render.
const MAIN_SUBMIT_SCOPE: NonNullable<SubmitPromptDeps['scope']> = {
  removeAttachments: attachments => mainComposerScope.removeOccurrences(attachments),
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
    runtimeIdByStoredSessionIdRef,
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
      // Images use their bounded base64 thumbnail so the optimistic bubble
      // renders inline without embedding the full source — see optimisticAttachmentRef.
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
      //
      // The guard reads the TARGET session's busy state (isTargetSessionBusy),
      // not the foreground flag: an explicit target (tile, queue drain) is
      // frequently not the session on screen, so the foreground flag would gate
      // one session's send on another session's turn.
      const hasSendable = Boolean(visibleText || terminalContextBlocks || attachments.length || hasImage)

      const guardSessionId = options?.sessionId ?? activeSessionIdRef.current

      if (
        !hasSendable ||
        (!options?.fromQueue && isTargetSessionBusy($sessionStates.get(), guardSessionId, busyRef.current))
      ) {
        return false
      }

      // Typing barge-in: a new send silences any in-flight spoken reply.
      if (isVoicePlaybackActive()) {
        markVoicePlaybackInterrupted()
        stopVoicePlayback()
      }

      // Barged mid-speech (here or via the voice loop's VAD)? Flag the submit
      // so the backend notes the interruption to the model.
      const interrupted = takeVoicePlaybackInterrupted()

      // Queue drains carry their source session explicitly. A background drain
      // must never inherit the currently selected session after the user moves
      // to another chat.
      const targetStoredSessionId = options?.storedSessionId ?? selectedStoredSessionIdRef.current

      const targetStartedInCurrentView =
        !targetStoredSessionId || targetStoredSessionId === selectedStoredSessionIdRef.current

      // A queued/background drain whose runtime binding was reaped must NOT
      // inherit the foreground runtime id when its storedSessionId targets a
      // different session — that would land the queued prompt in whichever
      // session the user happens to be viewing (cross-session leak). When the
      // drain is for the current view (no storedSessionId, or it matches the
      // foreground), the foreground runtime is correct and must be kept.
      const isBackgroundQueueDrain = Boolean(
        options?.fromQueue && options?.storedSessionId && options.storedSessionId !== selectedStoredSessionIdRef.current
      )

      let sessionId: null | string = options?.sessionId ?? (isBackgroundQueueDrain ? null : activeSessionIdRef.current)

      // A QUEUED runtime id is authoritative ONLY while it still belongs to its
      // stored session. On a session switch the composer's queue key flips with
      // the route while the foreground runtime id lags a resume behind, so a
      // drain can fire with storedSessionId=B but sessionId=A-runtime — and the
      // prompt.submit below would land B's queued prompt (and its whole answer
      // turn) inside A. Verify the pair against the central binding and drop a
      // stale queued id: the targetStoredSessionId resume path below then
      // rebinds the right runtime, exactly as a background drain with an
      // unknown binding does.
      //
      // Scoped to fromQueue on purpose. Only a drain pairs identifiers from two
      // different clocks; every other explicit-target caller resolves both ids
      // in the same tick and is authoritative by construction. A slash skill
      // dispatch into a fresh ⌘T tab (slash.ts) passes exactly this shape —
      // sessionId=tab-runtime, storedSessionId=tab-stored, no central binding
      // recorded yet — so an unscoped check would null the target and silently
      // drop the kickoff.
      //
      // The identity pair (storedSessionId === sessionId) is the fresh-chat
      // fallback — an unpersisted conversation's queue key IS its runtime id,
      // so it has no central binding to check against and is left untouched.
      if (
        options?.fromQueue &&
        options.sessionId &&
        options.storedSessionId &&
        options.storedSessionId !== options.sessionId
      ) {
        const boundRuntimeId = getRuntimeIdForStoredSession(options.storedSessionId)

        if (boundRuntimeId !== options.sessionId) {
          sessionId = boundRuntimeId
        }
      }

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
        targetStartedInCurrentView
          ? sessionContextDrift({
              startRouteToken: startingRouteToken,
              nowRouteToken: getRouteToken(),
              startSelectedStoredId: startingStoredSessionId,
              nowSelectedStoredId: selectedStoredSessionIdRef.current,
              submitTargetStoredId: startingStoredSessionId,
              composerScope: options?.composerScope,
              // The composer keys drafts/attachments on the durable lineage
              // root (survives auto-compression tip rotation), while
              // startingStoredSessionId is the live tip — resolve the target
              // into the same lineage-root domain before comparing, or every
              // submit into a session that has ever compressed would
              // false-positive-abort.
              submitTargetComposerScope: resolveComposerSessionKey(startingStoredSessionId, $sessions.get())
            })
          : null

      const targetIsCurrentView = (): boolean => targetStartedInCurrentView && !sessionDriftReason()

      // One submit in flight per session — drop any concurrent re-fire so a
      // stalled turn can't stack the same prompt into multiple real turns. The
      // foreground ChatBar and background drainers can briefly overlap during a
      // session switch; this per-session lock makes that safe.
      const submitLockKey = targetStoredSessionId || sessionId || startingActiveSessionId || '__pending_new__'

      if (!acquireSubmitInFlight(submitLockKey)) {
        return false
      }

      let submitLockReleased = false

      const releaseSubmitLock = () => {
        if (!submitLockReleased) {
          submitLockReleased = true
          releaseSubmitInFlight(submitLockKey)
        }
      }

      const optimisticId = `user-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`

      // What the bubble shows. A `/skill` send carries the whole expanded
      // skill body as its text — model-facing scaffolding — so the dispatcher
      // hands us the invocation to render instead. Everything else shows what
      // was typed.
      const bubbleText = options?.displayText ?? visibleText
      // Keep the user-send boundary stable when later ref resolution rewrites
      // the optimistic bubble in place.
      const submittedAt = Date.now() / 1000

      const buildUserMessage = (): ChatMessage => ({
        id: optimisticId,
        role: 'user',
        parts: [textPart(bubbleText || (attachmentRefs.length ? '' : attachments.map(a => a.label).join(', ')))],
        timestamp: submittedAt,
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
      const seedOptimistic = (sid: string) => {
        // Recents jump on send — not stream start, not turn resolve.
        const activity = bubbleText.trim() ? { preview: bubbleText.trim() } : undefined
        touchSessionActivity(sid, activity)

        if (targetStoredSessionId && targetStoredSessionId !== sid) {
          touchSessionActivity(targetStoredSessionId, activity)
        }

        updateSessionState(
          sid,
          state => ({
            ...state,
            // A fresh user message may never land after a still-pending
            // assistant bubble — settle any leftover (drop it when empty)
            // before appending, or a stale spinner gets stranded
            // mid-transcript above this message forever.
            messages: state.messages.some(m => m.id === optimisticId)
              ? state.messages
              : [...finalizeInterruptedMessages(state.messages, state.streamId), buildUserMessage()],
            busy: true,
            awaitingResponse: true,
            pendingBranchGroup: null,
            sawAssistantPayload: false,
            streamId: null,
            // Fresh submit = new turn — clear any leftover interrupt flag, else
            // mutateStream/completeAssistantMessage drop every delta of this turn
            // (what made drained-after-interrupt sends go silent).
            interrupted: false
          }),
          targetStoredSessionId
        )
      }

      // After sync rewrites refs, refresh the optimistic message in place so the
      // transcript shows the resolved @file: ref rather than the local path.
      const rewriteOptimistic = (sid: string) =>
        updateSessionState(
          sid,
          state => ({
            ...state,
            messages: state.messages.map(message => (message.id === optimisticId ? buildUserMessage() : message))
          }),
          targetStoredSessionId
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
          targetStoredSessionId
        )
      }

      const abortForSessionSwitch = (optimisticSessionId: null | string): false => {
        dropOptimistic(optimisticSessionId)
        releaseBusy()

        return false
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

      // Entry-time consistency check (#64789/#65328): activeSessionId is a
      // render-closure value that can already be stale relative to the
      // currently selected stored session by the time submit fires (e.g. a
      // fast reselect, or a new-chat draft's active ref not yet re-homed).
      // The #54527 drift guard only catches divergence that happens AFTER
      // this point, so an already-diverged runtime/stored pair sails
      // through it. Prove membership from BOTH directions against the same
      // cache rather than trusting an absent forward entry as "no
      // conflict" — a bare forward miss can't rule out the runtime being
      // known to belong to a DIFFERENT stored session (the failure mode a
      // one-directional check misses): if either direction disagrees,
      // activeSessionId is not trustworthy and the resume-by-stored-id path
      // below re-establishes the correct runtime id instead of silently
      // sending to the wrong one.
      const ownershipStoredSessionId = options?.sessionId ? null : targetStoredSessionId

      if (sessionId && ownershipStoredSessionId) {
        const provenRuntimeId = runtimeIdByStoredSessionIdRef.current.get(ownershipStoredSessionId)
        // A selected stored session requires positive ownership proof. A cache
        // miss is therefore unsafe too: the active runtime may belong to an
        // entirely different stored session, so resume the selected id instead
        // of sending to an unverified runtime.
        const knownMismatch = provenRuntimeId !== sessionId

        const runtimeOwnedByOtherStored = Array.from(runtimeIdByStoredSessionIdRef.current.entries()).some(
          ([storedId, runtimeId]) => runtimeId === sessionId && storedId !== ownershipStoredSessionId
        )

        if (knownMismatch || runtimeOwnedByOtherStored) {
          sessionId = null
        }
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

        const routedResumeDrift = sessionDriftReason()

        if (routedResumeDrift) {
          console.warn('[submit-drift-abort]', routedResumeDrift, { phase: 'post-routed-resume' })

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
          // Re-register on the session's OWNING profile — resuming on whichever
          // profile is live would fork the conversation into the wrong DB (#67603).
          const resumeProfile = await resolveSessionProfile(targetStoredSessionId)

          const resumed = await requestGateway<{ session_id: string }>('session.resume', {
            session_id: targetStoredSessionId,
            source: 'desktop',
            omit_messages: true,
            ...(resumeProfile ? { profile: resumeProfile } : {})
          })

          const resumeDrift = sessionDriftReason()

          if (resumeDrift) {
            console.warn('[submit-drift-abort]', resumeDrift, { phase: 'post-resume' })

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

        const resumeSettleDrift = sessionDriftReason()

        if (resumeSettleDrift) {
          console.warn('[submit-drift-abort]', resumeSettleDrift, { phase: 'post-resume-settle' })

          return abortForSessionSwitch(sessionId)
        }

        if (!sessionId) {
          return abortForSessionSwitch(null)
        }

        seedOptimistic(sessionId)
      }

      if (!sessionId) {
        try {
          sessionId = await createBackendSessionForSend(bubbleText)
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
          const createNullDrift = sessionDriftReason()

          if (createNullDrift) {
            console.warn('[submit-drift-abort]', createNullDrift, { phase: 'post-create-null' })

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
        // Attach runs BEFORE prompt.submit, so a stale runtime id fails there
        // first and submit's own recovery never runs — that asymmetry is why
        // plain text survived sleep/wake but images reported "session not
        // found". The attach path recovers and reports the live id back here.
        const attachResult = await syncAttachmentsForSubmit(sessionId, attachments, {
          updateComposerAttachments: usingComposerAttachments
        })

        const syncedAttachments = attachResult.attachments
        // Always a live string; pin it so TS narrows past the outer
        // `string | null` sessionId binding for prompt.submit.
        const liveSessionId = attachResult.sessionId

        sessionId = liveSessionId

        const attachmentsDrift = sessionDriftReason()

        if (attachmentsDrift) {
          console.warn('[submit-drift-abort]', attachmentsDrift, { phase: 'post-attachments' })

          return abortForSessionSwitch(liveSessionId)
        }

        // Rewrite the optimistic message + prompt text with the synced refs so
        // the gateway receives @file: paths that resolve in its workspace.
        // Images keep their inline bounded thumbnail — see optimisticAttachmentRef.
        attachmentRefs = syncedAttachments.map(optimisticAttachmentRef).filter((r): r is string => Boolean(r))
        rewriteOptimistic(liveSessionId)
        const text = buildContextText(syncedAttachments)

        const submitParams = (targetId: string) => ({
          session_id: targetId,
          text,
          ...(interrupted && { interrupted }),
          // Typed into the floating HUD, so the user is looking at another app
          // rather than at Hermes. The gateway turns this into a per-turn hint
          // to read the window underneath and work in it.
          ...($hudMode.get() && { surface: 'hud' }),
          // A queue drain is a "run after" message, never a live-turn
          // correction. The flag tells the gateway's busy path to hold it for
          // the next turn untouched — without it, losing the settle race
          // (client saw idle, server still unwinding) redirects or interrupts
          // the live turn with text the user explicitly queued.
          ...(options?.fromQueue && { queued: true })
        })

        // On sleep/wake the gateway's in-memory session may have been cleared
        // while the desktop app still holds the old session ID. The shared
        // resolver re-registers the stored session and retries once; every
        // other session-scoped RPC (attach, /compress, rewind, interrupt) goes
        // through the same helper so one policy covers the whole bug class.
        let submitErr: unknown = null

        try {
          const recoverStoredSessionId = targetStoredSessionId ?? selectedStoredSessionIdRef.current

          await withSessionNotFoundResume(
            sessionId,
            recoverStoredSessionId,
            liveId =>
              withSessionBusyRetry(() =>
                requestGateway('prompt.submit', submitParams(liveId), PROMPT_SUBMIT_REQUEST_TIMEOUT_MS)
              ),
            {
              requestGateway,
              driftReason: sessionDriftReason,
              onRecovered: recoveredId => {
                if (targetIsCurrentView()) {
                  activeSessionIdRef.current = recoveredId
                  setActiveSessionId(recoveredId)
                }
              }
            },
            // A starved backend loop (#55578 symptom d) rejects the submit even
            // though the stored session is fine — recover it like a dead id
            // instead of erroring out and losing the session binding.
            { alsoTimeout: true }
          )
        } catch (firstErr) {
          if (firstErr instanceof SessionRecoveryAborted) {
            console.warn('[submit-drift-abort]', firstErr.reason, { phase: 'post-resume-retry' })

            return abortForSessionSwitch(sessionId)
          }

          submitErr = firstErr
        }

        if (submitErr !== null) {
          throw submitErr
        }

        if (usingComposerAttachments) {
          // A submit owns only the occurrences that actually reached the
          // gateway. Tokenized chips match across staging clones; legacy chips
          // match by exact object identity, so a newer same-id replacement is
          // preserved while the staged object for a submitted file is removed.
          scope.removeAttachments(syncedAttachments)
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
        const occurredAt = Date.now() / 1000

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
                branchGroupId: state.pendingBranchGroup ?? undefined,
                completedAt: occurredAt,
                timestamp: occurredAt
              }
            ],
            busy: false,
            awaitingResponse: false,
            pendingBranchGroup: null,
            sawAssistantPayload: true
          }),
          targetStoredSessionId
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
      runtimeIdByStoredSessionIdRef,
      resumeStoredSession,
      scope,
      selectedStoredSessionIdRef,
      syncAttachmentsForSubmit,
      updateSessionState
    ]
  )
}

import type { PersistedTurn } from '@hermes/shared'
import { type MutableRefObject, useCallback } from 'react'

import { translateNow } from '@/i18n'
import {
  assistantTextPart,
  type ChatMessage,
  type ChatMessagePart,
  chatMessageText,
  completeOpenTimelineParts,
  mergeFinalAssistantText,
  renderMediaTags,
  sealOpenToolParts
} from '@/lib/chat-messages'
import type { ErrorSurface } from '@/lib/error-surface'
import { generatedImageEchoSources, stripGeneratedImageEchoes } from '@/lib/generated-images'
import { dispatchNativeNotification } from '@/store/native-notifications'
import { isDiskFullErrorMessage, notifyError } from '@/store/notifications'
import { broadcastTranscriptChanged } from '@/store/transcript-sync'

import { collapseDuplicateFinalAfterToolInterim, type DuplicateFinalCollapse } from './collapse-duplicate-final'
import { nextStreamMessageId, type UpdateSessionState } from './mutate-stream'
import { currentResponseParts, mergeCurrentResponseText } from './response-parts'
import { completionErrorText } from './utils'

export function useCompleteTurn(
  updateSessionState: UpdateSessionState,
  hydrateFromStoredSession: (
    attempts?: number,
    storedSessionId?: string | null,
    runtimeSessionId?: string | null
  ) => Promise<void>,
  scheduleSessionsRefresh: () => void,
  compactedTurnRef: MutableRefObject<Set<string>>
) {
  const finalizeInterimAssistantMessage = useCallback(
    (sessionId: string, text: string, occurredAt = Date.now() / 1000) => {
      updateSessionState(sessionId, state => {
        if (state.interrupted) {
          return state
        }

        const authoritativeText = renderMediaTags(text).trim()

        if (!authoritativeText) {
          return state
        }

        const streamId = state.streamId

        const replaceTextPart = (parts: ChatMessagePart[]) => {
          const visibleText = stripGeneratedImageEchoes(authoritativeText, generatedImageEchoSources(parts)).trim()

          // A later response can share this bubble after a suppressed interim.
          // A seal arriving after its tools, without any newer text, still
          // confirms the pre-tool response (legacy/delayed seal ordering).
          const hasNewResponse = currentResponseParts(parts).some(
            part => (part.type === 'text' || part.type === 'reasoning') && part.text.trim()
          )

          return hasNewResponse
            ? mergeCurrentResponseText(parts, visibleText, occurredAt)
            : mergeFinalAssistantText(parts, visibleText, occurredAt)
        }

        let nextMessages = state.messages

        if (streamId && nextMessages.some(m => m.id === streamId)) {
          // Seal the streaming bubble in place, marked interim so it renders
          // without an action footer (see ChatMessage.interim).
          nextMessages = nextMessages.map(m =>
            m.id === streamId
              ? {
                  ...m,
                  parts: completeOpenTimelineParts(replaceTextPart(m.parts), occurredAt),
                  completedAt: occurredAt,
                  pending: false,
                  interim: true
                }
              : m
          )
        } else {
          // No streaming bubble — create a standalone interim message
          nextMessages = [
            ...nextMessages,
            {
              id: nextStreamMessageId('assistant-interim'),
              role: 'assistant' as const,
              parts: [{ ...assistantTextPart(authoritativeText, occurredAt), completedAt: occurredAt }],
              timestamp: occurredAt,
              completedAt: occurredAt,
              pending: false,
              interim: true,
              branchGroupId: state.pendingBranchGroup ?? undefined
            }
          ]
        }

        return {
          ...state,
          messages: nextMessages,
          streamId: null,
          interimBoundaryPending: true,
          sawAssistantPayload: state.sawAssistantPayload || Boolean(authoritativeText)
        }
      })
    },
    [updateSessionState]
  )

  const completeAssistantMessage = useCallback(
    (
      sessionId: string,
      text: string,
      responsePreviewed?: boolean,
      failure?: { error: string; partial: boolean; surface?: ErrorSurface | null },
      occurredAt = Date.now() / 1000,
      persistedTurn?: PersistedTurn | null,
      responseTransformed?: boolean
    ) => {
      let shouldHydrate = false

      const completedState = updateSessionState(sessionId, state => {
        // Late completion from an already-cancelled turn: cancelRun has
        // already finalized the bubble (kept the partial text, dropped it if
        // empty). Re-running the dedupe below would replace the partial with
        // the just-cancelled full text, so we settle and bail instead.
        if (state.interrupted) {
          return {
            ...state,
            awaitingResponse: false,
            busy: false,
            needsInput: false,
            pendingBranchGroup: null,
            streamId: null,
            turnStartedAt: null,
            turnLive: false
          }
        }

        const streamId = state.streamId ?? state.heartbeatSettledStreamId ?? null
        const finalText = renderMediaTags(text).trim()
        // Structured failure from the terminal frame wins over the legacy text
        // heuristic ("Error: <provider detail>" texts don't match the regexes).
        const completionError = failure?.error ?? completionErrorText(finalText)
        // A partial failure's `text` is streamed output the user should keep,
        // not the error string — settle it like a normal reply AND mark the
        // bubble failed, instead of stripping the text.
        const keepFailedPartialText = Boolean(failure?.partial && finalText)
        const interimBoundaryPending = state.interimBoundaryPending

        // Wall-clock seconds this turn actually ran (message.start stamped
        // turnStartedAt). Read BEFORE the state return below nulls it.
        const durationS = state.turnStartedAt
          ? Math.max(1, Math.round((Date.now() - state.turnStartedAt) / 1000))
          : undefined

        const replaceTextPart = (parts: ChatMessagePart[], interim: boolean) => {
          const visibleFinalText = stripGeneratedImageEchoes(finalText, generatedImageEchoSources(parts)).trim()

          // Partial terminal errors carry the whole retained assistant buffer,
          // not just the response after the last tool (unlike healthy finals).
          return interim || keepFailedPartialText
            ? mergeFinalAssistantText(parts, visibleFinalText, occurredAt)
            : mergeCurrentResponseText(parts, visibleFinalText, occurredAt)
        }

        const withPersistedIdentity = (message: ChatMessage): ChatMessage => {
          const finalRowId = persistedTurn?.final_assistant_row_id
          const hasFinalRow = typeof finalRowId === 'number' && Number.isSafeInteger(finalRowId) && finalRowId > 0
          const finalPartIndex = message.parts.findLastIndex(part => part.type === 'text')

          return {
            ...message,
            durableComplete: persistedTurn?.complete === true,
            persistedTurn: persistedTurn ?? undefined,
            // A folded bubble can already address its first source row. Keep
            // that address and bind the final response's exact source as well.
            ...(hasFinalRow
              ? {
                  rowId: message.rowId ?? finalRowId,
                  parts: message.parts.map((part, index) =>
                    index === finalPartIndex ? { ...part, sourceRowId: finalRowId } : part
                  )
                }
              : {})
          }
        }

        // Settling the final response onto a bubble makes it the turn's real
        // reply — clear `interim` so it regains the action footer.
        const completeMessage = (message: ChatMessage): ChatMessage => {
          const settled = {
            ...message,
            completedAt: occurredAt,
            parts: completeOpenTimelineParts(message.parts, occurredAt),
            pending: false,
            interim: false,
            recovered: false,
            ...(durationS !== undefined ? { durationS } : {}),
            ...(completionError && failure?.surface ? { errorSurface: failure.surface } : {})
          }

          if (completionError && !keepFailedPartialText) {
            return withPersistedIdentity({
              ...settled,
              error: completionError,
              parts: settled.parts.filter(part => part.type !== 'text')
            })
          }

          return withPersistedIdentity({
            ...settled,
            parts: completeOpenTimelineParts(replaceTextPart(settled.parts, Boolean(message.interim)), occurredAt),
            ...(completionError ? { error: completionError } : {})
          })
        }

        const newAssistantFromCompletion = (): ChatMessage =>
          withPersistedIdentity({
            id: `assistant-${Date.now()}`,
            role: 'assistant',
            parts:
              completionError && !keepFailedPartialText
                ? []
                : [{ ...assistantTextPart(finalText, occurredAt), completedAt: occurredAt }],
            timestamp: occurredAt,
            completedAt: occurredAt,
            branchGroupId: state.pendingBranchGroup ?? undefined,
            ...(durationS !== undefined ? { durationS } : {}),
            ...(completionError && { error: completionError }),
            ...(completionError && failure?.surface ? { errorSurface: failure.surface } : {})
          })

        const prev = state.messages
        let nextMessages = prev

        // A new prompt or correction starts another occurrence, even when its
        // text (or answer) repeats. Hidden user rows are boundaries too.
        // A projected queued prompt is for the NEXT turn while this response
        // exists. message.start clears payload/interim ownership so a queued
        // turn completing without deltas still starts its own occurrence.
        const hasCurrentResponse = Boolean(streamId || state.sawAssistantPayload || interimBoundaryPending)

        const lastUserIndex = prev.findLastIndex(
          message => message.role === 'user' && !(hasCurrentResponse && message.id === `user-queued-${sessionId}`)
        )

        const streamIndex = streamId
          ? prev.findIndex((message, index) => index > lastUserIndex && message.id === streamId)
          : -1

        const settleAt = (index: number) =>
          prev.map((message, messageIndex) => (messageIndex === index ? completeMessage(message) : message))

        let collapsed: DuplicateFinalCollapse | null = null

        if (streamIndex >= 0) {
          collapsed = collapseDuplicateFinalAfterToolInterim(prev, streamIndex, {
            completeMessage,
            finalText,
            hasFailure: Boolean(failure) || Boolean(completionError),
            interimBoundaryPending
          })
          nextMessages = collapsed?.messages ?? settleAt(streamIndex)
        } else {
          const fallbackIndex = prev.findLastIndex(
            (message, index) => index > lastUserIndex && message.role === 'assistant' && !message.hidden
          )

          if (fallbackIndex >= 0) {
            const index = fallbackIndex
            const existing = prev[index]

            const existingText = chatMessageText({
              ...existing,
              parts: existing.interim || keepFailedPartialText ? existing.parts : currentResponseParts(existing.parts)
            }).trim()

            // The last assistant row is a sealed interim (a tool-call turn or a
            // verify-on-stop candidate — `message.interim` fires for BOTH, see
            // tui_gateway `_load_interim_assistant_messages`). When the final
            // completion is the SAME turn's reply, settle it onto that interim
            // instead of appending a second bubble. Continuity, not exact
            // equality: streaming can drop characters and the final may add a
            // trailing delta, so treat prefix-either-way as the same message.
            // (mergeFinalAssistantText, via completeMessage, does the real
            // text merge — replaces the interim's text with the full final.)
            const finalContinuesInterim = Boolean(
              existing.interim &&
              finalText &&
              existingText &&
              (finalText === existingText || finalText.startsWith(existingText) || existingText.startsWith(finalText))
            )

            if (existing.pending || (!interimBoundaryPending && finalText && existingText === finalText)) {
              nextMessages = settleAt(index)
            } else if (
              (interimBoundaryPending && (responsePreviewed || responseTransformed)) ||
              finalContinuesInterim
            ) {
              // Settle the interim in place instead of creating a duplicate —
              // the DB has one row, so the live UI must agree. Two distinct
              // settle paths with different boundary requirements:
              //
              // • responsePreviewed covers the verify-on-stop continuation-
              //   budget case, where the final may be a rewrite sharing no
              //   prefix with the interim. Because there is no continuity
              //   guarantee, it must stay gated on the session's
              //   `interimBoundaryPending` flag: after a new `message.start`
              //   resets the flag, a previewed final is a DISTINCT reply and
              //   must append its own bubble, never overwrite the interim
              //   (otherwise interim('old') → message.start →
              //   complete({response_previewed: true, text: 'new'}) would
              //   silently destroy 'old').
              //
              // • responseTransformed (a transform_llm_output hook rewrote the
              //   final after streaming, e.g. pseudonym restore) shares the
              //   same no-continuity shape, so it takes the same boundary gate.
              //
              // • finalContinuesInterim (prefix-either-way continuity, same
              //   text or one a prefix of the other) is safe to settle
              //   flag-free within this user occurrence: a `message.start`
              //   reset between this turn's interim and completion must not
              //   force an append of a duplicate bubble (#74560). This also
              //   closes the non-previewed tool-call gap from #63679.
              nextMessages = settleAt(index)
            } else if (finalText) {
              nextMessages = [...prev, newAssistantFromCompletion()]
            }
          } else {
            // Nothing streamed after the boundary and no `message.start` since
            // the seal: a completion whose text IS the sealed pre-boundary reply
            // is that reply's own completion (a redirect rejected after the row
            // was painted, or a steer the model absorbed without new output),
            // not a second occurrence. Anything else respects the boundary.
            const sealedIndex =
              !streamId && interimBoundaryPending && finalText
                ? prev.findLastIndex(
                    (message, index) => index < lastUserIndex && message.role === 'assistant' && !message.hidden
                  )
                : -1

            const sealed = sealedIndex >= 0 ? prev[sealedIndex] : null

            if (sealed?.interim === true && chatMessageText(sealed).trim() === finalText) {
              nextMessages = settleAt(sealedIndex)
            } else if (finalText) {
              nextMessages = [...prev, newAssistantFromCompletion()]
            }
          }
        }

        // Turn-settle reconciliation: a `tool.complete` event lost to a
        // degraded websocket leaves its tool row spinning forever. The turn is
        // provably done here — nothing can still be running — so seal any
        // tool-call parts that never saw their completion event.
        nextMessages = sealOpenToolParts(nextMessages)

        const hasInlineError = nextMessages.some(
          (m, index) => index > lastUserIndex && m.role === 'assistant' && m.error && !m.hidden
        )

        const lastVisible = [...nextMessages].reverse().find(m => !m.hidden)
        const unresolvedUserTail = lastVisible?.role === 'user'

        const sameTurnId = collapsed?.keptId ?? (streamIndex >= 0 ? streamId : null)

        const sameTurnAssistant = sameTurnId
          ? nextMessages.find(m => m.id === sameTurnId)
          : nextMessages.findLast((m, index) => index > lastUserIndex && m.role === 'assistant' && !m.hidden)

        const localVisibleText = sameTurnAssistant ? chatMessageText(sameTurnAssistant).trim() : ''
        // Having streamed the reply normally means this window owns the whole
        // turn and re-reading stored history would be wasted work. That only
        // holds for a turn it STARTED: an adopted one (resumed onto a session
        // already running elsewhere) arrives reply-first, with no prompt row,
        // so it has to hydrate or the user's own message never shows up.
        // Adopted turns still hydrate so a resume-onto-running session can
        // pick up the user's prompt row — unless this window already has
        // visible assistant text and the terminal frame is empty. In that
        // case hydrate would replace the live bubble with a stored empty
        // row (#95514; adoptedRunningTurn must not short-circuit).
        shouldHydrate =
          !completionError &&
          !hasInlineError &&
          // A visible user message with no reply after the terminal frame
          // means this window never rendered the turn's output. When the
          // frame also carries no text, the reply only exists in stored
          // history — hydrate to catch up instead of leaving the transcript
          // blank until restart (#88036). A non-empty frame still settles
          // locally, so the user-tail guard keeps applying there.
          (!unresolvedUserTail || !finalText) &&
          !(localVisibleText && !finalText) &&
          (state.adoptedRunningTurn || !state.sawAssistantPayload)

        return {
          ...state,
          messages: nextMessages,
          adoptedRunningTurn: false,
          heartbeatSettledStreamId: null,
          streamId: null,
          pendingBranchGroup: null,
          awaitingResponse: false,
          busy: false,
          needsInput: false,
          interimBoundaryPending: false,
          turnStartedAt: null,
          turnLive: false
        }
      })

      // Persistence / mid-turn disk-full failures land as a terminal frame with
      // an error string, not a rejected prompt.submit. Toast them here so a
      // full disk never looks like a silent no-reply. Only fire on actual
      // failure signals — never on a healthy reply that happens to say
      // "disk full".
      const diskFullSignal = failure?.error || (failure ? text : '')

      if (diskFullSignal && isDiskFullErrorMessage(diskFullSignal)) {
        notifyError(new Error(diskFullSignal), translateNow('notifications.errors.diskFull'))
      }

      scheduleSessionsRefresh()

      if (completedState.storedSessionId) {
        broadcastTranscriptChanged({
          messageCount: completedState.messages.length,
          sessionId: completedState.storedSessionId
        })
      }

      if (compactedTurnRef.current.delete(sessionId)) {
        shouldHydrate = false
      }

      if (shouldHydrate) {
        void hydrateFromStoredSession(3, completedState.storedSessionId, sessionId)
      }

      dispatchNativeNotification({
        body: text.slice(0, 140) || translateNow('notifications.native.turnDoneBody'),
        kind: 'turnDone',
        sessionId,
        title: translateNow('notifications.native.turnDoneTitle')
      })
    },
    [compactedTurnRef, hydrateFromStoredSession, scheduleSessionsRefresh, updateSessionState]
  )

  const failAssistantMessage = useCallback(
    (sessionId: string, errorMessage: string, occurredAt = Date.now() / 1000, surface?: ErrorSurface | null) => {
      updateSessionState(sessionId, state => {
        const streamId = state.streamId ?? `assistant-error-${Date.now()}`
        const groupId = state.pendingBranchGroup ?? undefined
        const prev = state.messages
        const error = errorMessage.trim() || 'Hermes reported an error'
        // The `error` event carries no descriptor; the dispatcher may recover
        // one from the text (SESSION_NOT_OWNED, disk_full) so the card gates
        // its buttons like a classified turn.
        const errorSurface = surface ? { errorSurface: surface } : {}

        const durationS = state.turnStartedAt
          ? Math.max(1, Math.round((Date.now() - state.turnStartedAt) / 1000))
          : undefined

        const nextMessages = prev.some(m => m.id === streamId)
          ? prev.map(message =>
              message.id === streamId
                ? {
                    ...message,
                    completedAt: occurredAt,
                    error,
                    ...errorSurface,
                    parts: completeOpenTimelineParts(message.parts, occurredAt),
                    pending: false,
                    ...(durationS !== undefined ? { durationS } : {})
                  }
                : message
            )
          : [
              ...prev,
              {
                id: streamId,
                role: 'assistant' as const,
                parts: [],
                timestamp: occurredAt,
                completedAt: occurredAt,
                error,
                ...errorSurface,
                pending: false,
                branchGroupId: groupId,
                ...(durationS !== undefined ? { durationS } : {})
              }
            ]

        return {
          ...state,
          messages: nextMessages,
          streamId: null,
          pendingBranchGroup: null,
          sawAssistantPayload: true,
          awaitingResponse: false,
          busy: false,
          needsInput: false,
          interimBoundaryPending: false,
          turnStartedAt: null,
          turnLive: false
        }
      })
    },
    [updateSessionState]
  )

  return { finalizeInterimAssistantMessage, completeAssistantMessage, failAssistantMessage }
}

/**
 * Prompt actions for a SESSION TILE — the same verbs the primary chat wires
 * (submit incl. slash, cancel, steer, edit, reload, restore, branch-hide
 * sync), targeted at the tile's session instead of the active one. State
 * writes go through the delegate's `updateSession` (the wiring cache), so
 * the cache, the primary view, and every tile mirror stay one truth; view
 * concerns (busy pill, transcript) reach the tile via its `$sessionStates`
 * slice — never the global `$busy`/`$messages`.
 */

import type { AppendMessage, ThreadMessage } from '@assistant-ui/react'
import { useCallback, useMemo, useRef } from 'react'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import type { ClientSessionState } from '@/app/types'
import { useI18n } from '@/i18n'
import { textPart } from '@/lib/chat-messages'
import { SLASH_COMMAND_RE } from '@/lib/chat-runtime'
import { triggerHaptic } from '@/lib/haptics'
import { clearClarifyRequest } from '@/store/clarify'
import type { ComposerAttachment } from '@/store/composer'
import { resetSessionBackground } from '@/store/composer-status'
import { notifyError } from '@/store/notifications'
import { clearPreviewArtifacts } from '@/store/preview-status'
import { clearAllPrompts } from '@/store/prompts'
import { $connection, $sessions, sessionMatchesStoredId } from '@/store/session'
import { $sessionStates, sessionTileDelegate } from '@/store/session-states'
import { broadcastSessionsChanged } from '@/store/session-sync'
import { clearSessionSubagents } from '@/store/subagents'
import { clearSessionTodos } from '@/store/todos'
import { setSessionDraftingTool } from '@/store/tool-drafting'
import type { SessionInfo } from '@/types/hermes'

import { uploadComposerAttachment } from '../session/hooks/use-prompt-actions'
import {
  appendMidTurnUserMessage,
  applyBranchVisibility,
  applyReloadOptimistic,
  applyRewindOptimistic,
  finalizeInterruptedMessages,
  planEdit,
  planReload,
  planRestore,
  rebindSurvivorRowIds,
  runRewindSubmit,
  type SurvivorUserRowIds
} from '../session/hooks/use-prompt-actions/rewind'
import { useSubmitPrompt } from '../session/hooks/use-prompt-actions/submit'
import {
  markSessionRecentlyInterrupted,
  shouldInterruptBeforeRewind,
  type SubmitTextOptions,
  withSessionNotFoundResume
} from '../session/hooks/use-prompt-actions/utils'
import { upsertOptimisticSession } from '../session/hooks/use-session-actions/utils'

import type { ComposerScope } from './composer/scope'

/**
 * List a tile's session in the sidebar/tab strip on its first send.
 *
 * A ⌘T tab's session is created UNLISTED (see `openNewSessionTile`), so it has
 * no `$sessions` row until its first turn persists and a refresh surfaces it —
 * for that whole first exchange the tab and the sidebar read "New session".
 * ⌘N has no such gap: its session is created per-send and seeded with the
 * user's text as the row preview. Seeding the same way here names the session
 * within the first message; the server's auto-title supersedes it once the turn
 * completes.
 *
 * No-ops on empty text and on a session that is already listed, so re-sends
 * never clobber a real title with a raw message preview.
 */
export function listTileSessionRow(deps: {
  cwd?: string
  model?: string
  preview: string
  runtimeId: string
  sessions: readonly SessionInfo[]
  storedSessionId: string
}): boolean {
  const preview = deps.preview.trim()

  if (!preview || deps.sessions.some(session => sessionMatchesStoredId(session, deps.storedSessionId))) {
    return false
  }

  upsertOptimisticSession(
    { info: { cwd: deps.cwd, model: deps.model }, session_id: deps.runtimeId, stored_session_id: deps.storedSessionId },
    deps.storedSessionId,
    null,
    preview
  )
  broadcastSessionsChanged()

  return true
}

interface SessionTileActionsArgs {
  runtimeId: string
  scope: ComposerScope
  storedSessionId: string
}

export function useSessionTileActions({ runtimeId, scope, storedSessionId }: SessionTileActionsArgs) {
  const { t } = useI18n()
  const copy = t.desktop
  const { requestGateway } = useGatewayRequest()

  const runtimeIdRef = useRef(runtimeId)
  runtimeIdRef.current = runtimeId
  const storedIdRef = useRef(storedSessionId)
  storedIdRef.current = storedSessionId
  // A tile IS its session (see the comment on the useSubmitPrompt call below)
  // A tile owns one stable stored/runtime pair, so seed the shared ownership
  // cache explicitly rather than relying on the primary route cache.
  const runtimeIdByStoredSessionIdRef = useRef(new Map([[storedSessionId, runtimeId]]))
  runtimeIdByStoredSessionIdRef.current.set(storedSessionId, runtimeId)

  // Tile busy tracks the SESSION state, never the global $busy — and it must
  // read LIVE. A render-time snapshot goes stale (this hook's host doesn't
  // re-render on busy edges), and a stale `true` silently blocks every
  // subsequent submit ("tile only sends one message"). The setter is a no-op:
  // session state owns busy; submit's optimistic writes flow through
  // updateSession.
  const busyRef = useMemo(
    () =>
      ({
        get current() {
          return $sessionStates.get()[runtimeIdRef.current]?.busy ?? false
        },
        set current(_value: boolean) {
          // Owned by session state.
        }
      }) as { current: boolean },
    []
  )

  const update = useCallback(
    (updater: (state: ClientSessionState) => ClientSessionState) =>
      sessionTileDelegate()?.updateSession(runtimeIdRef.current, updater),
    []
  )

  const readState = useCallback(() => $sessionStates.get()[runtimeIdRef.current], [])
  const readMessages = useCallback(() => readState()?.messages ?? [], [readState])

  // A ⌘T tab's session is unlisted until its first turn persists — seed the
  // row from the user's first message so the tab and sidebar name it right
  // away (see listTileSessionRow).
  const listTileSession = useCallback((preview: string) => {
    const runtimeId = runtimeIdRef.current
    const state = $sessionStates.get()[runtimeId]

    listTileSessionRow({
      cwd: state?.cwd,
      model: state?.model,
      preview,
      runtimeId,
      sessions: $sessions.get(),
      storedSessionId: storedIdRef.current
    })
  }, [])

  // Tile-side attachment staging: same upload rules as the primary submit
  // (skip synced/pathless, byte-upload files+images), against the tile scope.
  const syncAttachmentsForSubmit = useCallback(
    async (
      sessionId: string,
      attachments: ComposerAttachment[],
      options: { updateComposerAttachments?: boolean } = {}
    ): Promise<{ attachments: ComposerAttachment[]; sessionId: string }> => {
      const remote = $connection.get()?.mode === 'remote'
      let liveSessionId = sessionId
      const synced: ComposerAttachment[] = []

      // A tile owns its own runtime binding, so a recovery here rebinds the
      // tile's ref rather than the foreground session's.
      const onSessionRecovered = (recoveredId: string) => {
        liveSessionId = recoveredId
        runtimeIdRef.current = recoveredId
      }

      for (const attachment of attachments) {
        if (!attachment.path || attachment.attachedSessionId === liveSessionId) {
          synced.push(attachment)

          continue
        }

        if (attachment.kind === 'image' || attachment.kind === 'file') {
          const next = await uploadComposerAttachment(attachment, {
            backendCwd: readState()?.cwd,
            remote,
            requestGateway,
            sessionId: liveSessionId,
            storedSessionId: storedIdRef.current,
            onSessionRecovered
          })

          if (options.updateComposerAttachments ?? true) {
            if (attachment.occurrenceId) {
              // Merge staging into the latest state for this exact tile-chip
              // occurrence. A preview may complete while upload is pending,
              // and remove + same-path reattach must not receive stale staging.
              scope.attachments.updateIfCurrent(attachment, {
                attachedSessionId: next.attachedSessionId,
                label: next.label,
                path: next.path,
                refText: next.refText,
                uploadState: next.uploadState
              })
            } else {
              scope.attachments.update(next)
            }
          }

          synced.push(next)

          continue
        }

        synced.push(attachment)
      }

      return { attachments: synced, sessionId: liveSessionId }
    },
    [requestGateway, scope.attachments]
  )

  // The REAL submit pipeline with tile seams: session always exists, and the
  // scope's writers replace the global view/attachment writes.
  const submitPromptText = useSubmitPrompt({
    activeSessionIdRef: runtimeIdRef,
    busyRef,
    copy,
    createBackendSessionForSend: async () => runtimeIdRef.current,
    getRoutedStoredSessionId: () => storedIdRef.current,
    getRuntimeIdForStoredSession: storedId => (storedId === storedIdRef.current ? runtimeIdRef.current : null),
    // A tile IS its session — no route to abandon, so the create-abort guard's
    // token is a stable constant (the guard never trips for a tile).
    getRouteToken: () => runtimeId,
    requestGateway,
    runtimeIdByStoredSessionIdRef,
    // Tile ids are always bound before this hook mounts, so routed recovery is
    // unreachable here; keep the shared submit contract explicit.
    resumeStoredSession: () => undefined,
    selectedStoredSessionIdRef: storedIdRef,
    syncAttachmentsForSubmit,
    updateSessionState: (sessionId, updater) => sessionTileDelegate()!.updateSession(sessionId, updater),
    scope: {
      removeAttachments: attachments => scope.attachments.removeOccurrences(attachments),
      readAttachments: () => scope.attachments.$attachments.get(),
      // Busy/messages flow through updateSession -> the tile's state slice;
      // the primary view atoms must never see a tile turn.
      setAwaitingResponse: () => undefined,
      setBusy: () => undefined,
      setMessages: () => undefined
    }
  })

  const submitText = useCallback(
    async (rawText: string, options?: SubmitTextOptions) => {
      const visibleText = rawText.trim()
      const attachments = options?.attachments ?? scope.attachments.$attachments.get()

      listTileSession(visibleText)

      if (!attachments.length && SLASH_COMMAND_RE.test(visibleText)) {
        triggerHaptic('selection')
        await sessionTileDelegate()?.executeSlash(visibleText, runtimeIdRef.current)

        return true
      }

      return await submitPromptText(rawText, options)
    },
    [listTileSession, scope.attachments.$attachments, submitPromptText]
  )

  const cancelRun = useCallback(async () => {
    const sessionId = runtimeIdRef.current

    // Frontend busy clears immediately; gateway wind-down can lag (#83855).
    markSessionRecentlyInterrupted(sessionId)

    update(state => ({
      ...state,
      messages: finalizeInterruptedMessages(state.messages, state.streamId),
      busy: false,
      awaitingResponse: false,
      streamId: null,
      pendingBranchGroup: null,
      needsInput: false,
      interrupted: true
    }))

    clearSessionTodos(sessionId)
    clearSessionSubagents(sessionId)
    resetSessionBackground(sessionId)
    setSessionDraftingTool(sessionId, '')
    clearAllPrompts(sessionId)
    clearClarifyRequest(undefined, sessionId)

    try {
      await withSessionNotFoundResume(
        sessionId,
        storedIdRef.current,
        liveId => requestGateway('session.interrupt', { session_id: liveId }),
        {
          requestGateway,
          onRecovered: recoveredId => {
            runtimeIdRef.current = recoveredId
          }
        }
      )
    } catch (err) {
      notifyError(err, copy.stopFailed)
    }
  }, [copy.stopFailed, requestGateway, update])

  const steerPrompt = useCallback(
    async (rawText: string): Promise<boolean> => {
      const text = rawText.trim()
      const sessionId = runtimeIdRef.current

      if (!text || !sessionId) {
        return false
      }

      const messageId = `user-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`

      const mutate = (updater: (state: ClientSessionState) => ClientSessionState) =>
        sessionTileDelegate()?.updateSession(sessionId, updater)

      // Match the primary composer: record the correction in arrival order —
      // sealed already-streamed output above, correction below, post-redirect
      // deltas below that — before awaiting the redirect RPC, whose completion
      // can race us. The old insert-before-the-active-reply splice put the
      // bubble above output the user had already read (#73793), and its
      // last-assistant fallback could land it mid-thread when the stream id
      // was missing or stale (#83151).
      mutate(state =>
        appendMidTurnUserMessage(state, {
          id: messageId,
          role: 'user' as const,
          parts: [textPart(text)]
        })
      )

      const discardOptimisticMessage = () =>
        mutate(state => ({
          ...state,
          messages: state.messages.filter(message => message.id !== messageId)
        }))

      const moveOptimisticMessageToEnd = () =>
        mutate(state => {
          const message = state.messages.find(candidate => candidate.id === messageId)

          return message
            ? { ...state, messages: [...state.messages.filter(candidate => candidate.id !== messageId), message] }
            : state
        })

      try {
        const { result } = await withSessionNotFoundResume(
          sessionId,
          storedIdRef.current,
          liveId => requestGateway<{ status?: string }>('session.redirect', { session_id: liveId, text }),
          {
            requestGateway,
            onRecovered: recoveredId => {
              runtimeIdRef.current = recoveredId
            }
          }
        )

        if (result?.status === 'redirected') {
          triggerHaptic('submit')

          return true
        }

        if (result?.status === 'queued') {
          moveOptimisticMessageToEnd()
          triggerHaptic('submit')

          return true
        }
      } catch {
        discardOptimisticMessage()
        // Swallow — the caller queues the text so nothing is lost.

        return false
      }

      discardOptimisticMessage()

      return false
    },
    [requestGateway]
  )

  // Rewind primitive (interrupt-first for live turns, busy-retry) — shared with
  // the primary chat so the two can't diverge.
  const submitRewind = useCallback(
    (
      text: string,
      truncateOrdinal: number | undefined,
      interruptFirst: boolean,
      truncateMessageId?: string,
      truncateRowId?: number,
      sourceText?: string
    ) =>
      runRewindSubmit(
        requestGateway,
        runtimeIdRef.current,
        text,
        truncateOrdinal,
        truncateMessageId,
        interruptFirst,
        {
          storedSessionId: storedIdRef.current,
          onSessionRecovered: recoveredId => {
            runtimeIdRef.current = recoveredId
          }
        },
        truncateRowId,
        sourceText
      ),
    [requestGateway]
  )

  // After a durable rewind the surviving bubbles' cached rowIds are stale (the
  // gateway re-inserted the kept prefix as new SQLite rows). Rebind them to the
  // authoritative post-rewrite ids so the NEXT rewind/edit/regenerate doesn't
  // send a dead id and get refused with 4018 (consecutive-rewind staleness,
  // #83202 review).
  const applySurvivorRowIds = useCallback(
    (survivorRowIds: SurvivorUserRowIds | undefined) => {
      if (!survivorRowIds) {
        return
      }

      update(state => ({ ...state, messages: rebindSurvivorRowIds(state.messages, survivorRowIds) }))
    },
    [update]
  )

  const reloadFromMessage = useCallback(
    async (parentId: string | null) => {
      const state = readState()

      if (!state || state.busy) {
        return
      }

      const plan = planReload(state.messages, parentId)

      if (!plan) {
        return
      }

      update(current => applyReloadOptimistic(current, plan))

      try {
        // Recovery for a dead runtime id rides inside submitRewind →
        // runRewindSubmit (withSessionNotFoundResume + runtime rebind), so the
        // PR-era inline prompt.submit wrapper is superseded on current main.
        applySurvivorRowIds(
          await submitRewind(
            plan.text,
            plan.truncateOrdinal,
            false,
            plan.truncateMessageId,
            plan.truncateRowId,
            plan.sourceText
          )
        )
      } catch (err) {
        update(current => ({ ...current, busy: false, awaitingResponse: false, turnLive: false, turnStartedAt: null }))
        notifyError(err, copy.regenerateFailed)
      }
    },
    [applySurvivorRowIds, copy.regenerateFailed, readState, submitRewind, update]
  )

  const restoreToMessage = useCallback(
    async (messageId: string, target?: { text?: string; userOrdinal?: number | null }) => {
      const sessionId = runtimeIdRef.current
      const messages = readMessages()
      const plan = planRestore(messages, messageId, target)

      clearSessionTodos(sessionId)
      resetSessionBackground(sessionId)
      clearPreviewArtifacts(sessionId)

      const interruptFirst = shouldInterruptBeforeRewind({
        busy: readState()?.busy ?? false,
        sessionId
      })

      update(state => applyRewindOptimistic(state, plan.sourceIndex))

      try {
        applySurvivorRowIds(
          await submitRewind(
            plan.text,
            plan.truncateOrdinal,
            interruptFirst,
            plan.truncateMessageId,
            plan.truncateRowId,
            plan.sourceText
          )
        )
      } catch (err) {
        update(state => ({ ...state, busy: false, awaitingResponse: false, turnLive: false, turnStartedAt: null, messages }))
        throw err
      }
    },
    [applySurvivorRowIds, readMessages, readState, submitRewind, update]
  )

  const editMessage = useCallback(
    async (edited: AppendMessage) => {
      const messages = readMessages()
      const plan = planEdit(messages, edited)

      if (!plan) {
        return
      }

      const sessionId = runtimeIdRef.current

      clearSessionTodos(sessionId)
      resetSessionBackground(sessionId)
      clearPreviewArtifacts(sessionId)

      const interruptFirst = shouldInterruptBeforeRewind({
        busy: readState()?.busy ?? false,
        sessionId
      })

      update(state => applyRewindOptimistic(state, plan.sourceIndex, plan.editedMessage))

      try {
        applySurvivorRowIds(
          await submitRewind(
            plan.text,
            plan.truncateOrdinal,
            interruptFirst,
            plan.truncateMessageId,
            plan.truncateRowId,
            plan.sourceText
          )
        )
      } catch (err) {
        update(state => ({ ...state, busy: false, awaitingResponse: false, turnLive: false, turnStartedAt: null, messages }))
        notifyError(err, copy.editFailed)
      }
    },
    [applySurvivorRowIds, copy.editFailed, readMessages, readState, submitRewind, update]
  )

  // Branch-visibility sync (assistant-ui hides non-active branches).
  const handleThreadMessagesChange = useCallback(
    (nextMessages: readonly ThreadMessage[]) => update(state => applyBranchVisibility(state, nextMessages)),
    [update]
  )

  const dismissError = useCallback(
    (messageId: string) => {
      update(state => ({ ...state, messages: state.messages.filter(m => m.id !== messageId) }))
    },
    [update]
  )

  return useMemo(
    () => ({
      cancelRun,
      dismissError,
      editMessage,
      handleThreadMessagesChange,
      reloadFromMessage,
      restoreToMessage,
      steerPrompt,
      submitText
    }),
    [
      cancelRun,
      dismissError,
      editMessage,
      handleThreadMessagesChange,
      reloadFromMessage,
      restoreToMessage,
      steerPrompt,
      submitText
    ]
  )
}

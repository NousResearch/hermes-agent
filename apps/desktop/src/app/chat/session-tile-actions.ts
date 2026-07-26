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
import { PROMPT_SUBMIT_REQUEST_TIMEOUT_MS } from '@/hermes'
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
import type { SessionInfo } from '@/types/hermes'

import { uploadComposerAttachment } from '../session/hooks/use-prompt-actions'
import {
  applyBranchVisibility,
  applyReloadOptimistic,
  applyRewindOptimistic,
  finalizeInterruptedMessages,
  planEdit,
  planReload,
  planRestore,
  runRewindSubmit,
  truncateSubmitParams
} from '../session/hooks/use-prompt-actions/rewind'
import { useSubmitPrompt } from '../session/hooks/use-prompt-actions/submit'
import { type SubmitTextOptions } from '../session/hooks/use-prompt-actions/utils'
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
    ): Promise<ComposerAttachment[]> => {
      const remote = $connection.get()?.mode === 'remote'
      const synced: ComposerAttachment[] = []

      for (const attachment of attachments) {
        if (!attachment.path || attachment.attachedSessionId === sessionId) {
          synced.push(attachment)

          continue
        }

        if (attachment.kind === 'image' || attachment.kind === 'file') {
          const next = await uploadComposerAttachment(attachment, { remote, requestGateway, sessionId })

          if (options.updateComposerAttachments ?? true) {
            scope.attachments.update(next)
          }

          synced.push(next)

          continue
        }

        synced.push(attachment)
      }

      return synced
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
    // Tile ids are always bound before this hook mounts, so routed recovery is
    // unreachable here; keep the shared submit contract explicit.
    resumeStoredSession: () => undefined,
    selectedStoredSessionIdRef: storedIdRef,
    syncAttachmentsForSubmit,
    updateSessionState: (sessionId, updater) => sessionTileDelegate()!.updateSession(sessionId, updater),
    scope: {
      clearAttachments: scope.attachments.clear,
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

  const appendSystemNote = useCallback(
    (text: string) => {
      update(state => ({
        ...state,
        messages: [
          ...state.messages,
          {
            id: `system-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
            role: 'system',
            parts: [textPart(text)]
          }
        ]
      }))
    },
    [update]
  )

  const cancelRun = useCallback(async () => {
    const sessionId = runtimeIdRef.current

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
    clearAllPrompts(sessionId)
    clearClarifyRequest(undefined, sessionId)

    try {
      await requestGateway('session.interrupt', { session_id: sessionId })
    } catch (err) {
      notifyError(err, copy.stopFailed)
    }
  }, [copy.stopFailed, requestGateway, update])

  const steerPrompt = useCallback(
    async (rawText: string): Promise<boolean> => {
      const text = rawText.trim()

      if (!text) {
        return false
      }

      try {
        const result = await requestGateway<{ status?: string }>('session.steer', {
          session_id: runtimeIdRef.current,
          text
        })

        if (result?.status === 'queued') {
          triggerHaptic('submit')
          appendSystemNote(`steer:${text}`)

          return true
        }
      } catch {
        // Swallow — the caller queues the text so nothing is lost.
      }

      return false
    },
    [appendSystemNote, requestGateway]
  )

  // Rewind primitive (interrupt-first for live turns, busy-retry) — shared with
  // the primary chat so the two can't diverge.
  const submitRewind = useCallback(
    (text: string, truncateOrdinal: number | undefined, interruptFirst: boolean) =>
      runRewindSubmit(requestGateway, runtimeIdRef.current, text, truncateOrdinal, interruptFirst),
    [requestGateway]
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
        await requestGateway(
          'prompt.submit',
          {
            session_id: runtimeIdRef.current,
            text: plan.text,
            ...truncateSubmitParams(plan.truncateOrdinal)
          },
          PROMPT_SUBMIT_REQUEST_TIMEOUT_MS
        )
      } catch (err) {
        update(current => ({ ...current, busy: false, awaitingResponse: false }))
        notifyError(err, copy.regenerateFailed)
      }
    },
    [copy.regenerateFailed, readState, requestGateway, update]
  )

  const restoreToMessage = useCallback(
    async (messageId: string, target?: { text?: string; userOrdinal?: number | null }) => {
      const sessionId = runtimeIdRef.current
      const messages = readMessages()
      const plan = planRestore(messages, messageId, target)

      clearSessionTodos(sessionId)
      resetSessionBackground(sessionId)
      clearPreviewArtifacts(sessionId)

      const wasBusy = readState()?.busy ?? false

      update(state => applyRewindOptimistic(state, plan.sourceIndex))

      try {
        await submitRewind(plan.text, plan.truncateOrdinal, wasBusy)
      } catch (err) {
        update(state => ({ ...state, busy: false, awaitingResponse: false, messages }))
        throw err
      }
    },
    [readMessages, readState, submitRewind, update]
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

      const wasBusy = readState()?.busy ?? false

      update(state => applyRewindOptimistic(state, plan.sourceIndex, plan.editedMessage))

      try {
        await submitRewind(plan.text, plan.truncateOrdinal, wasBusy)
      } catch (err) {
        update(state => ({ ...state, busy: false, awaitingResponse: false, messages }))
        notifyError(err, copy.editFailed)
      }
    },
    [copy.editFailed, readMessages, readState, submitRewind, update]
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

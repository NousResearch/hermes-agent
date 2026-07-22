import type { AppendMessage, ThreadMessage } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { type MutableRefObject, useCallback, useEffect, useRef } from 'react'

import { PROMPT_SUBMIT_REQUEST_TIMEOUT_MS, transcribeAudio } from '@/hermes'
import { useI18n } from '@/i18n'
import { stripAnsi } from '@/lib/ansi'
import { type ChatMessage, textPart } from '@/lib/chat-messages'
import { pathLabel, SLASH_COMMAND_RE } from '@/lib/chat-runtime'
import { sanitizeComposerInput } from '@/lib/composer-input-sanitize'
import { triggerHaptic } from '@/lib/haptics'
import { setMutableRef } from '@/lib/mutable-ref'
import { normalize } from '@/lib/text'
import { clearClarifyRequest } from '@/store/clarify'
import {
  $composerAttachments,
  type ComposerAttachment,
  setComposerAttachmentUploadState,
  updateComposerAttachment
} from '@/store/composer'
import { resetSessionBackground } from '@/store/composer-status'
import { clearNotifications, notify, notifyError } from '@/store/notifications'
import { clearPreviewArtifacts } from '@/store/preview-status'
import { clearAllPrompts } from '@/store/prompts'
import {
  $busy,
  $connection,
  $messages,
  setAwaitingResponse,
  setBusy,
  setMessages,
  setTurnStartedAt
} from '@/store/session'
import { clearSessionSubagents } from '@/store/subagents'
import { clearSessionTodos } from '@/store/todos'

import type {
  ClientSessionState,
  FileAttachResponse,
  HandoffFailResponse,
  HandoffRequestResponse,
  HandoffStateResponse,
  ImageAttachResponse,
  SessionRedirectResponse
} from '../../../types'

import {
  applyBranchVisibility,
  applyReloadOptimistic,
  applyRewindOptimistic,
  finalizeInterruptedMessages,
  planEdit,
  planReload,
  planRestore,
  runRewindSubmit
} from './rewind'
import { useSlashCommand } from './slash'
import { useSubmitPrompt } from './submit'
import {
  blobToDataUrl,
  delay,
  friendlyRemoteAttachError,
  type GatewayRequest,
  inlineErrorMessage,
  isSessionNotFoundError,
  readFileDataUrlForAttach,
  readImageForRemoteAttach,
  type SubmitTextOptions
} from './utils'

interface HandoffResult {
  ok: boolean
  error?: string
}

/**
 * Stage one file/image attachment into the session workspace and return the
 * attachment rewritten with the gateway-side ref. Images upload their bytes in
 * remote mode (so vision works) and pass the path locally; non-image files
 * upload bytes remotely and pass the path locally. Throws on failure so callers
 * can surface an error. Shared by submit-time sync, the eager drop-time upload,
 * and the message-edit composer drop — keep them in lockstep.
 */
export async function uploadComposerAttachment(
  attachment: ComposerAttachment,
  opts: { remote: boolean; requestGateway: GatewayRequest; sessionId: string }
): Promise<ComposerAttachment> {
  const { remote, requestGateway, sessionId } = opts
  const path = attachment.path ?? ''
  const label = attachment.label || pathLabel(path)

  if (attachment.kind === 'image') {
    let result: ImageAttachResponse

    if (remote) {
      let payload: Awaited<ReturnType<typeof readImageForRemoteAttach>>

      try {
        payload = await readImageForRemoteAttach(path)
      } catch (err) {
        throw friendlyRemoteAttachError(err, label)
      }

      if (!payload) {
        throw new Error(`Could not read ${label}`)
      }

      result = await requestGateway<ImageAttachResponse>('image.attach_bytes', {
        session_id: sessionId,
        content_base64: payload.contentBase64,
        filename: payload.filename
      })
    } else {
      result = await requestGateway<ImageAttachResponse>('image.attach', {
        path,
        session_id: sessionId
      })
    }

    if (!result.attached) {
      throw new Error(result.message || `Could not attach ${label}`)
    }

    const attachedPath = result.path || path

    return {
      ...attachment,
      attachedSessionId: sessionId,
      label: attachedPath ? pathLabel(attachedPath) : attachment.label,
      path: attachedPath,
      uploadState: undefined
    }
  }

  // Non-image file.
  let dataUrl: string | null = null

  if (remote) {
    try {
      dataUrl = await readFileDataUrlForAttach(path)
    } catch (err) {
      throw friendlyRemoteAttachError(err, label)
    }

    if (!dataUrl) {
      throw new Error(`Could not read ${label}`)
    }
  }

  const result = await requestGateway<FileAttachResponse>('file.attach', {
    name: label,
    path,
    session_id: sessionId,
    ...(dataUrl ? { data_url: dataUrl } : {})
  })

  if (!result.attached || !result.ref_text) {
    throw new Error(result.message || `Could not attach ${label}`)
  }

  return {
    ...attachment,
    attachedSessionId: sessionId,
    refText: result.ref_text,
    uploadState: undefined
  }
}

interface PromptActionsOptions {
  activeSessionId: string | null
  activeSessionIdRef: MutableRefObject<string | null>
  busyRef: MutableRefObject<boolean>
  branchCurrentSession: () => Promise<boolean>
  createBackendSessionForSend: (preview?: string | null) => Promise<string | null>
  getRoutedStoredSessionId: () => null | string
  getRuntimeIdForStoredSession: (storedSessionId: string) => null | string
  getRouteToken: () => string
  handleSkinCommand: (arg: string) => string
  openMemoryGraph: () => void
  refreshSessions: () => Promise<void>
  requestGateway: <T>(method: string, params?: Record<string, unknown>, timeoutMs?: number) => Promise<T>
  resumeStoredSession: (storedSessionId: string) => Promise<void> | void
  selectedStoredSessionIdRef: MutableRefObject<string | null>
  startFreshSessionDraft: () => void
  sttEnabled: boolean
  updateSessionState: (
    sessionId: string,
    updater: (state: ClientSessionState) => ClientSessionState,
    storedSessionId?: string | null
  ) => ClientSessionState
}

/** Everything a slash handler needs about the invocation it's serving. */

interface RestoreMessageTarget {
  text?: string
  userOrdinal?: number | null
}

export function usePromptActions({
  activeSessionId,
  activeSessionIdRef,
  busyRef,
  branchCurrentSession,
  createBackendSessionForSend,
  getRoutedStoredSessionId,
  getRuntimeIdForStoredSession,
  getRouteToken,
  handleSkinCommand,
  openMemoryGraph,
  refreshSessions,
  requestGateway,
  resumeStoredSession,
  selectedStoredSessionIdRef,
  startFreshSessionDraft,
  sttEnabled,
  updateSessionState
}: PromptActionsOptions) {
  const { t } = useI18n()
  const copy = t.desktop

  const appendSessionTextMessage = useCallback(
    (sessionId: string, role: ChatMessage['role'], text: string) => {
      // Strip ANSI: slash-command output from the backend worker carries SGR
      // color codes (e.g. "Unknown command" in red). The ESC byte is invisible
      // in the chat panel, so without this the `[1;31m…[0m` payload leaks as
      // literal text.
      const body = stripAnsi(text).trim()

      if (!body) {
        return
      }

      updateSessionState(
        sessionId,
        state => ({
          ...state,
          messages: [
            ...state.messages,
            {
              id: `${role}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
              role,
              parts: [textPart(body)]
            }
          ]
        }),
        selectedStoredSessionIdRef.current
      )
    },
    [selectedStoredSessionIdRef, updateSessionState]
  )

  // In-flight drop-time eager uploads, keyed by attachment id. Submit joins
  // these before re-uploading so a drop-then-immediately-Enter can't fire
  // file.attach twice and stage duplicate copies on the gateway.
  const eagerUploadInFlight = useRef<Map<string, Promise<void>>>(new Map())

  const syncAttachmentsForSubmit = useCallback(
    async (
      sessionId: string,
      attachments: ComposerAttachment[],
      options: { updateComposerAttachments?: boolean } = {}
    ): Promise<ComposerAttachment[]> => {
      const updateComposerAttachments = options.updateComposerAttachments ?? true
      const remote = $connection.get()?.mode === 'remote'
      const synced: ComposerAttachment[] = []

      for (const original of attachments) {
        let attachment = original

        // Join a drop-time eager upload still in flight for this attachment
        // before deciding anything — otherwise submit and the eager task both
        // call file.attach and stage duplicate files. After it settles, take the
        // store's updated copy (its gateway ref, or its failure) over the stale
        // pre-upload snapshot.
        const inFlight = eagerUploadInFlight.current.get(attachment.id)

        if (inFlight) {
          await inFlight
          attachment = $composerAttachments.get().find(item => item.id === attachment.id) ?? attachment
        }

        // Already-synced or pathless refs (terminal, url, etc.) pass through.
        // A drop-time eager upload may already have staged this one (matching
        // attachedSessionId) — don't re-upload it.
        if (!attachment.path || attachment.attachedSessionId === sessionId) {
          synced.push(attachment)

          continue
        }

        if (attachment.kind === 'image' || attachment.kind === 'file') {
          const nextAttachment = await uploadComposerAttachment(attachment, { remote, requestGateway, sessionId })

          // Update-only: never resurrect a chip the user removed mid-upload.
          if (updateComposerAttachments) {
            updateComposerAttachment(nextAttachment)
          }

          synced.push(nextAttachment)

          continue
        }

        synced.push(attachment)
      }

      return synced
    },
    [requestGateway]
  )

  // Stage a freshly dropped file as soon as it lands (when a session already
  // exists), so the upload runs while the user is still typing rather than
  // stalling the send. The card shows a spinner via `uploadState`; on success
  // the chip carries its gateway-side ref so submit skips re-uploading.
  //
  // Images are intentionally NOT eager-uploaded: attachImagePath adds the chip
  // and then fills in `previewUrl` (the base64 thumbnail) on a second tick, so
  // an eager upload would race that write — clobbering the thumbnail and
  // swapping `path` to a gateway path the local preview can't read. Images are
  // small and still byte-upload at submit via image.attach_bytes.
  const eagerlyUploadAttachment = useCallback(
    async (sessionId: string, attachment: ComposerAttachment) => {
      const remote = $connection.get()?.mode === 'remote'

      setComposerAttachmentUploadState(attachment.id, 'uploading')

      try {
        // Update-only: if the user removed the chip while this was uploading,
        // don't resurrect it — just drop the staged result on the floor.
        updateComposerAttachment(await uploadComposerAttachment(attachment, { remote, requestGateway, sessionId }))
      } catch (err) {
        // Leave the chip in place so submit-time sync can retry (or the user can
        // remove it) and flag the card; also toast so a hard failure (unreadable
        // file, gateway perms) isn't swallowed while the user keeps typing.
        setComposerAttachmentUploadState(attachment.id, 'error')
        notifyError(err, copy.dropFiles)
      }
    },
    [copy.dropFiles, requestGateway]
  )

  const composerAttachments = useStore($composerAttachments)

  useEffect(() => {
    if (!activeSessionId) {
      return
    }

    for (const attachment of composerAttachments) {
      const needsUpload =
        attachment.kind === 'file' &&
        Boolean(attachment.path) &&
        !attachment.attachedSessionId &&
        !attachment.uploadState &&
        !eagerUploadInFlight.current.has(attachment.id)

      if (!needsUpload) {
        continue
      }

      const task = eagerlyUploadAttachment(activeSessionId, attachment).finally(() =>
        eagerUploadInFlight.current.delete(attachment.id)
      )

      eagerUploadInFlight.current.set(attachment.id, task)
    }
  }, [activeSessionId, composerAttachments, eagerlyUploadAttachment])

  const submitPromptText = useSubmitPrompt({
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
    updateSessionState
  })

  // Queue a handoff of this session to a messaging platform and watch it to
  // a terminal state. We only write the request through the gateway; the
  // separate `hermes gateway` process performs the actual transfer, so we
  // poll `handoff.state` (mirror of the CLI's block-poll) for the result.
  const handoffSession = useCallback(
    async (
      platform: string,
      options?: { onProgress?: (state: string) => void; sessionId?: string }
    ): Promise<HandoffResult> => {
      const sid = options?.sessionId || activeSessionIdRef.current

      if (!sid) {
        return { error: copy.sessionUnavailable, ok: false }
      }

      const target = normalize(platform)

      if (!target) {
        return { error: copy.handoff.failed(''), ok: false }
      }

      try {
        options?.onProgress?.('pending')
        await requestGateway<HandoffRequestResponse>('handoff.request', {
          platform: target,
          session_id: sid
        })
      } catch (err) {
        return { error: inlineErrorMessage(err, copy.handoff.failed(target)), ok: false }
      }

      const markCompleted = (): HandoffResult => {
        appendSessionTextMessage(sid, 'system', copy.handoff.systemNote(target))
        notify({ kind: 'success', message: copy.handoff.success(target) })

        return { ok: true }
      }

      const deadline = Date.now() + 60_000
      let lastState = 'pending'

      while (Date.now() < deadline) {
        await delay(800)

        let record: HandoffStateResponse

        try {
          record = await requestGateway<HandoffStateResponse>('handoff.state', { session_id: sid })
        } catch {
          continue
        }

        const state = record.state || 'pending'

        if (state !== lastState) {
          options?.onProgress?.(state)
          lastState = state
        }

        if (state === 'completed') {
          return markCompleted()
        }

        if (state === 'failed') {
          return { error: record.error || copy.handoff.failed(target), ok: false }
        }
      }

      const cleanup = await requestGateway<HandoffFailResponse>('handoff.fail', {
        error: copy.handoff.timedOut,
        session_id: sid
      }).catch(() => null)

      if (cleanup?.state === 'completed') {
        return markCompleted()
      }

      return { error: copy.handoff.timedOut, ok: false }
    },
    [activeSessionIdRef, appendSessionTextMessage, copy, requestGateway]
  )

  const executeSlashCommand = useSlashCommand({
    activeSessionIdRef,
    appendSessionTextMessage,
    branchCurrentSession,
    busyRef,
    copy,
    createBackendSessionForSend,
    handleSkinCommand,
    handoffSession,
    openMemoryGraph,
    refreshSessions,
    requestGateway,
    resumeStoredSession,
    startFreshSessionDraft,
    submitPromptText
  })

  const submitText = useCallback(
    async (rawText: string, options?: SubmitTextOptions) => {
      const visibleText = sanitizeComposerInput(rawText).trim()
      const attachments = options?.attachments ?? $composerAttachments.get()

      if (!attachments.length && SLASH_COMMAND_RE.test(visibleText)) {
        triggerHaptic('selection')
        await executeSlashCommand(visibleText)

        return true
      }

      return await submitPromptText(rawText, options)
    },
    [executeSlashCommand, submitPromptText]
  )

  const transcribeVoiceAudio = useCallback(
    async (audio: Blob) => {
      if (!sttEnabled) {
        throw new Error(copy.sttDisabled)
      }

      const dataUrl = await blobToDataUrl(audio)
      const result = await transcribeAudio(dataUrl, audio.type)

      return result.transcript
    },
    [copy.sttDisabled, sttEnabled]
  )

  const cancelRun = useCallback(async () => {
    // Read from the ref, not the closure-captured `activeSessionId`. The
    // actions bag is a stable ref mutated in place (Object.assign on each
    // ContribWiring render), and ChatRoutesSurface is memoized on that stable
    // ref — so it does NOT re-render when activeSessionId changes, which means
    // the ChatView element's onCancel prop holds a stale cancelRun closure.
    // The closure's `activeSessionId` can be a previous session's id (or null
    // from a new-chat draft), sending session.interrupt to the wrong session.
    // The ref is updated via useEffect on every activeSessionId change, so it
    // always reflects the current session — same pattern submitText uses.
    const sessionId = activeSessionIdRef.current

    const releaseBusy = () => {
      setMutableRef(busyRef, false)
      setBusy(false)
    }

    setAwaitingResponse(false)
    setTurnStartedAt(null)

    if (!sessionId) {
      releaseBusy()
      setMessages(finalizeInterruptedMessages($messages.get()))

      return
    }

    updateSessionState(sessionId, state => {
      const streamId = state.streamId
      const messages = finalizeInterruptedMessages(state.messages, streamId)

      return {
        ...state,
        messages,
        busy: false,
        awaitingResponse: false,
        streamId: null,
        pendingBranchGroup: null,
        needsInput: false,
        interrupted: true,
        turnStartedAt: null
      }
    })

    clearSessionTodos(sessionId)
    clearSessionSubagents(sessionId)
    resetSessionBackground(sessionId)
    // Stop ends the turn, so the gateway is no longer blocked on any prompt it
    // raised. Drop this session's pending clarify / approval / sudo / secret so
    // a dead panel (and the sidebar "needs input" dot) can't linger and accept
    // an answer the backend will reject.
    clearAllPrompts(sessionId)
    clearClarifyRequest(undefined, sessionId)

    try {
      await requestGateway('session.interrupt', { session_id: sessionId })
      releaseBusy()
    } catch (err) {
      let stopError = err

      if (isSessionNotFoundError(err) && selectedStoredSessionIdRef.current) {
        try {
          const resumed = await requestGateway<{ session_id: string }>('session.resume', {
            session_id: selectedStoredSessionIdRef.current,
            source: 'desktop'
          })

          const recoveredId = resumed?.session_id

          if (recoveredId) {
            activeSessionIdRef.current = recoveredId
            await requestGateway('session.interrupt', { session_id: recoveredId })
            releaseBusy()

            return
          }
        } catch (resumeErr) {
          stopError = resumeErr
        }
      }

      releaseBusy()
      notifyError(stopError, copy.stopFailed)
    }
  }, [activeSessionIdRef, busyRef, copy.stopFailed, requestGateway, selectedStoredSessionIdRef, updateSessionState])

  // The desktop steering action is an immediate correction: the core cancels
  // model generation and rebuilds the live turn with displayed reasoning and
  // completed work intact. During a tool it waits for the safe result boundary.
  // Returns false when the turn raced to completion so the composer can queue.
  const redirectPrompt = useCallback(
    async (rawText: string): Promise<boolean> => {
      const text = sanitizeComposerInput(rawText).trim()
      const sessionId = activeSessionId || activeSessionIdRef.current

      if (!text || !sessionId) {
        return false
      }

      // Accepted whether the live turn was redirected in place or queued for
      // the next turn (the build window, before the agent is wired) — either
      // way the correction reaches the model, so record it once as a real user
      // message after the interrupted checkpoint, matching the durable core
      // transcript rather than a system note that changes role after reload.
      const send = async (id: string): Promise<boolean> => {
        const result = await requestGateway<SessionRedirectResponse>('session.redirect', { session_id: id, text })

        if (result?.status === 'redirected' || result?.status === 'queued') {
          triggerHaptic('submit')
          appendSessionTextMessage(id, 'user', text)

          return true
        }

        return false
      }

      try {
        return await send(sessionId)
      } catch (err) {
        // A stale runtime id after reconnect 404s ("session not found"): resume
        // the stored session and retry once, mirroring stopPrompt so a
        // correction right after a reconnect isn't lost to the race.
        if (isSessionNotFoundError(err) && selectedStoredSessionIdRef.current) {
          try {
            const resumed = await requestGateway<{ session_id: string }>('session.resume', {
              session_id: selectedStoredSessionIdRef.current,
              source: 'desktop'
            })

            const recoveredId = resumed?.session_id

            if (recoveredId) {
              activeSessionIdRef.current = recoveredId

              return await send(recoveredId)
            }
          } catch {
            // fall through — caller queues so nothing is lost
          }
        }
        // Swallow — caller queues the text so nothing is lost.
      }

      return false
    },
    [activeSessionId, activeSessionIdRef, appendSessionTextMessage, requestGateway, selectedStoredSessionIdRef]
  )

  const reloadFromMessage = useCallback(
    async (parentId: string | null) => {
      if (!activeSessionId || $busy.get()) {
        return
      }

      const plan = planReload($messages.get(), parentId)

      if (!plan) {
        return
      }

      clearNotifications()
      updateSessionState(activeSessionId, state => applyReloadOptimistic(state, plan))

      try {
        await requestGateway(
          'prompt.submit',
          { session_id: activeSessionId, text: plan.text, truncate_before_user_ordinal: plan.truncateOrdinal },
          PROMPT_SUBMIT_REQUEST_TIMEOUT_MS
        )
      } catch (err) {
        updateSessionState(activeSessionId, state => ({
          ...state,
          busy: false,
          awaitingResponse: false
        }))
        notifyError(err, copy.regenerateFailed)
      }
    },
    [activeSessionId, copy.regenerateFailed, requestGateway, updateSessionState]
  )

  // Cursor-style "restore checkpoint": rewind the conversation to a past user
  // prompt and run it again from there. Reuses the edit composer's rewind
  // mechanism — `prompt.submit` with `truncate_before_user_ordinal` drops that
  // user turn and everything after it from the session history, then the same
  // text is submitted as a fresh turn. Callers confirm before invoking; errors
  // are rethrown so callers can surface failures. Idle rewinds submit directly:
  // interrupting an idle agent can leave a stale interrupt flag that cancels the
  // fresh turn. Live/stuck turns interrupt first, and a raced "session busy"
  // response interrupts + retries through the shared busy gate.
  const submitRewindPrompt = useCallback(
    (sessionId: string, text: string, truncateOrdinal: number | undefined, interruptFirst: boolean) =>
      runRewindSubmit(requestGateway, sessionId, text, truncateOrdinal, interruptFirst),
    [requestGateway]
  )

  const restoreToMessage = useCallback(
    async (messageId: string, target?: RestoreMessageTarget) => {
      const sessionId = activeSessionId || activeSessionIdRef.current

      if (!sessionId) {
        throw new Error('No active session to restore.')
      }

      const messages = $messages.get()
      const plan = planRestore(messages, messageId, target)

      // The turns we're discarding may have spawned todos and background
      // processes; they belong to the abandoned timeline, so wipe their status
      // rows (and kill the live processes) before the fresh run repopulates.
      clearSessionTodos(sessionId)
      resetSessionBackground(sessionId)
      clearPreviewArtifacts(sessionId)

      clearNotifications()
      setMutableRef(busyRef, true)
      setBusy(true)
      setAwaitingResponse(true)
      updateSessionState(sessionId, state => applyRewindOptimistic(state, plan.sourceIndex))

      try {
        await submitRewindPrompt(sessionId, plan.text, plan.truncateOrdinal, busyRef.current || $busy.get())
      } catch (err) {
        // The rewind never landed (e.g. the gateway stayed busy past the retry
        // deadline). Roll the optimistic truncation back to the full original
        // history so the UI doesn't desync from what's persisted — leaving it
        // truncated is what made subsequent sends look duplicative.
        setMutableRef(busyRef, false)
        setBusy(false)
        setAwaitingResponse(false)
        updateSessionState(sessionId, state => ({
          ...state,
          busy: false,
          awaitingResponse: false,
          messages
        }))
        throw err
      }
    },
    [activeSessionId, activeSessionIdRef, busyRef, submitRewindPrompt, updateSessionState]
  )

  const editMessage = useCallback(
    async (edited: AppendMessage) => {
      const sessionId = activeSessionId || activeSessionIdRef.current
      const messages = $messages.get()
      const plan = sessionId ? planEdit(messages, edited) : null

      if (!sessionId || !plan) {
        return
      }

      // Sending an edit is a revert: rewind to this prompt and re-run with the
      // new text (submitRewindPrompt interrupts a live turn first). Same as
      // restore, so drop the abandoned timeline's todos/background rows before
      // the re-run repopulates them.
      clearSessionTodos(sessionId)
      resetSessionBackground(sessionId)
      clearPreviewArtifacts(sessionId)

      clearNotifications()
      setMutableRef(busyRef, true)
      setBusy(true)
      setAwaitingResponse(true)
      updateSessionState(sessionId, state => applyRewindOptimistic(state, plan.sourceIndex, plan.editedMessage))

      const isStaleTargetError = (err: unknown) =>
        /no longer in session history|not in session history/i.test(err instanceof Error ? err.message : String(err))

      try {
        await submitRewindPrompt(sessionId, plan.text, plan.truncateOrdinal, busyRef.current || $busy.get())
      } catch (err) {
        let surfaced = err

        if (!plan.isFailedTurn && isStaleTargetError(err)) {
          try {
            // Already interrupted on the first attempt — submit as a plain resend.
            await submitRewindPrompt(sessionId, plan.text, undefined, false)

            return
          } catch (retryErr) {
            surfaced = retryErr
          }
        }

        // Roll the optimistic edit/truncation back to the original history so the
        // UI stays in sync with what's persisted instead of stranding a partial
        // timeline.
        setMutableRef(busyRef, false)
        setBusy(false)
        setAwaitingResponse(false)
        updateSessionState(sessionId, state => ({ ...state, busy: false, awaitingResponse: false, messages }))
        notifyError(surfaced, copy.editFailed)
      }
    },
    [activeSessionId, activeSessionIdRef, busyRef, copy.editFailed, submitRewindPrompt, updateSessionState]
  )

  const handleThreadMessagesChange = useCallback(
    (nextMessages: readonly ThreadMessage[]) => {
      const sessionId = activeSessionIdRef.current

      if (!sessionId) {
        return
      }

      updateSessionState(sessionId, state => applyBranchVisibility(state, nextMessages))
    },
    [activeSessionIdRef, updateSessionState]
  )

  return {
    cancelRun,
    editMessage,
    // Session tiles route their slash input here (targets THEIR session via
    // options.sessionId; app-level effects — branch, handoff — act on main).
    executeSlashCommand,
    handleThreadMessagesChange,
    handoffSession,
    reloadFromMessage,
    restoreToMessage,
    redirectPrompt,
    /** @deprecated Use `redirectPrompt` — this is an active-turn redirect, not tool steer. */
    steerPrompt: redirectPrompt,
    submitText,
    transcribeVoiceAudio
  }
}

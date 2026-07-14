import type { AppendMessage, ThreadMessage } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { type MutableRefObject, useCallback, useEffect, useRef } from 'react'

import { PROMPT_SUBMIT_REQUEST_TIMEOUT_MS, transcribeAudio } from '@/hermes'
import { useI18n } from '@/i18n'
import { stripAnsi } from '@/lib/ansi'
import { branchGroupForUser, type ChatMessage, chatMessageText, textPart } from '@/lib/chat-messages'
import { pathLabel, SLASH_COMMAND_RE } from '@/lib/chat-runtime'
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
import { $busy, $connection, $messages, setActiveSessionId, setAwaitingResponse, setBusy, setMessages } from '@/store/session'
import { clearSessionSubagents } from '@/store/subagents'
import { clearSessionTodos } from '@/store/todos'

import type {
  ClientSessionState,
  FileAttachResponse,
  HandoffFailResponse,
  HandoffRequestResponse,
  HandoffStateResponse,
  ImageAttachResponse,
  SessionSteerResponse
} from '../../../types'

import { useSlashCommand } from './slash'
import { useSubmitPrompt } from './submit'
import {
  appendText,
  blobToDataUrl,
  delay,
  friendlyRemoteAttachError,
  type GatewayRequest,
  inlineErrorMessage,
  isSessionBusyError,
  isSessionNotFoundError,
  readFileDataUrlForAttach,
  readImageForRemoteAttach,
  type SubmitTextOptions,
  visibleUserIndexAtOrdinal,
  visibleUserOrdinal,
  withSessionBusyRetry
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
          appendSessionTextMessage(sid, 'system', copy.handoff.systemNote(target))
          notify({ kind: 'success', message: copy.handoff.success(target) })

          return { ok: true }
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
        appendSessionTextMessage(sid, 'system', copy.handoff.systemNote(target))
        notify({ kind: 'success', message: copy.handoff.success(target) })

        return { ok: true }
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
      const visibleText = rawText.trim()
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
    const sessionId = activeSessionId || activeSessionIdRef.current

    const releaseBusy = () => {
      setMutableRef(busyRef, false)
      setBusy(false)
    }

    setAwaitingResponse(false)

    const finalizeMessages = (messages: ChatMessage[], streamId?: string | null) =>
      messages
        .filter(message => !((message.pending || message.id === streamId) && !chatMessageText(message).trim()))
        .map(message => (message.pending || message.id === streamId ? { ...message, pending: false } : message))

    if (!sessionId) {
      releaseBusy()
      setMessages(finalizeMessages($messages.get()))

      return
    }

    updateSessionState(sessionId, state => {
      const streamId = state.streamId
      const messages = finalizeMessages(state.messages, streamId)

      return {
        ...state,
        messages,
        busy: false,
        awaitingResponse: false,
        streamId: null,
        pendingBranchGroup: null,
        needsInput: false,
        interrupted: true
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
            setActiveSessionId(recoveredId)
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
  }, [
    activeSessionId,
    activeSessionIdRef,
    busyRef,
    copy.stopFailed,
    requestGateway,
    selectedStoredSessionIdRef,
    updateSessionState
  ])

  // Steer = nudge the live turn without interrupting: the gateway appends the
  // text to the next tool result so the model reads it on its next iteration
  // (desktop parity with `/steer`). Returns false on reject (no live tool
  // window) so the caller can fall back to queueing the words for the next turn.
  const steerPrompt = useCallback(
    async (rawText: string): Promise<boolean> => {
      const text = rawText.trim()
      const sessionId = activeSessionId || activeSessionIdRef.current

      if (!text || !sessionId) {
        return false
      }

      try {
        const result = await requestGateway<SessionSteerResponse>('session.steer', { session_id: sessionId, text })

        if (result?.status === 'queued') {
          triggerHaptic('submit')
          // Inline note (not a toast) so the nudge lives in the transcript next
          // to the turn it steered. The `steer:` prefix is rendered as a codicon
          // row by SystemMessage (see STEER_NOTE_RE), same style as slash output.
          appendSessionTextMessage(sessionId, 'system', `steer:${text}`)

          return true
        }
      } catch {
        // Swallow — caller queues the text so nothing is lost.
      }

      return false
    },
    [activeSessionId, activeSessionIdRef, appendSessionTextMessage, requestGateway]
  )

  const reloadFromMessage = useCallback(
    async (parentId: string | null) => {
      if (!activeSessionId || $busy.get()) {
        return
      }

      const messages = $messages.get()
      const parentIndex = parentId ? messages.findIndex(message => message.id === parentId) : messages.length - 1

      const userIndex =
        parentIndex >= 0
          ? [...messages.slice(0, parentIndex + 1)].reverse().findIndex(message => message.role === 'user')
          : -1

      if (userIndex < 0) {
        return
      }

      const absoluteUserIndex = parentIndex - userIndex
      const userMessage = messages[absoluteUserIndex]
      const userText = userMessage ? chatMessageText(userMessage).trim() : ''

      if (!userText) {
        return
      }

      const targetAssistant =
        parentId && messages[parentIndex]?.role === 'assistant'
          ? messages[parentIndex]
          : messages.slice(absoluteUserIndex + 1).find(message => message.role === 'assistant')

      const branchGroupId = targetAssistant?.branchGroupId ?? branchGroupForUser(userMessage)
      const truncateBeforeUserOrdinal = visibleUserOrdinal(messages, absoluteUserIndex)

      clearNotifications()
      updateSessionState(activeSessionId, state => {
        const nextUserIndex = state.messages.findIndex(
          (message, index) => index > absoluteUserIndex && message.role === 'user'
        )

        const end = nextUserIndex < 0 ? state.messages.length : nextUserIndex

        return {
          ...state,
          busy: true,
          awaitingResponse: true,
          pendingBranchGroup: branchGroupId,
          sawAssistantPayload: false,
          interrupted: false,
          messages: [
            ...state.messages.slice(0, absoluteUserIndex + 1),
            ...state.messages
              .slice(absoluteUserIndex + 1, end)
              .map(message => (message.role === 'assistant' ? { ...message, branchGroupId, hidden: true } : message))
          ]
        }
      })

      try {
        await requestGateway(
          'prompt.submit',
          {
            session_id: activeSessionId,
            text: userText,
            truncate_before_user_ordinal: truncateBeforeUserOrdinal
          },
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
    async (sessionId: string, text: string, truncateOrdinal: number | undefined, interruptFirst: boolean) => {
      const interrupt = async () => {
        try {
          await requestGateway('session.interrupt', { session_id: sessionId })
        } catch {
          // Best-effort. The submit path still gates on the gateway state.
        }
      }

      const submit = () =>
        requestGateway(
          'prompt.submit',
          {
            session_id: sessionId,
            text,
            ...(truncateOrdinal !== undefined && { truncate_before_user_ordinal: truncateOrdinal })
          },
          PROMPT_SUBMIT_REQUEST_TIMEOUT_MS
        )

      if (interruptFirst) {
        await interrupt()
      }

      try {
        await submit()
      } catch (err) {
        if (!isSessionBusyError(err)) {
          throw err
        }

        await interrupt()
        await withSessionBusyRetry(submit)
      }
    },
    [requestGateway]
  )

  const restoreToMessage = useCallback(
    async (messageId: string, target?: RestoreMessageTarget) => {
      const sessionId = activeSessionId || activeSessionIdRef.current

      if (!sessionId) {
        throw new Error('No active session to restore.')
      }

      const messages = $messages.get()
      const idIndex = messages.findIndex(m => m.id === messageId && m.role === 'user')

      const fallbackIndex =
        target?.userOrdinal === null || target?.userOrdinal === undefined
          ? -1
          : visibleUserIndexAtOrdinal(messages, target.userOrdinal)

      const sourceIndex = idIndex >= 0 ? idIndex : fallbackIndex
      const source = messages[sourceIndex]

      if (!source || source.role !== 'user') {
        throw new Error('Could not find the message to restore.')
      }

      const text = (chatMessageText(source).trim() || target?.text?.trim() || '').trim()

      if (!text) {
        throw new Error('Cannot restore an empty message.')
      }

      const truncateBeforeUserOrdinal =
        target?.userOrdinal === null || target?.userOrdinal === undefined
          ? visibleUserOrdinal(messages, sourceIndex)
          : target.userOrdinal

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
      updateSessionState(sessionId, state => ({
        ...state,
        busy: true,
        awaitingResponse: true,
        pendingBranchGroup: null,
        sawAssistantPayload: false,
        interrupted: false,
        messages: state.messages.slice(0, sourceIndex + 1)
      }))

      try {
        await submitRewindPrompt(sessionId, text, truncateBeforeUserOrdinal, busyRef.current || $busy.get())
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
      const sourceId = edited.sourceId || edited.parentId
      const text = appendText(edited)

      if (!sessionId || !sourceId || !text || edited.role !== 'user') {
        return
      }

      const messages = $messages.get()
      const sourceIndex = messages.findIndex(m => m.id === sourceId)
      const source = messages[sourceIndex]

      if (!source || source.role !== 'user' || chatMessageText(source).trim() === text) {
        return
      }

      // Sending an edit is a revert: rewind to this prompt and re-run with the
      // new text. It can fire mid-turn; submitRewindPrompt always interrupts
      // first, so a live turn is wound down before the resubmit.

      // Failed turn: optimistic user msg never reached the gateway, so truncating
      // by ordinal would 422. Submit as a plain resend instead.
      const nextMessage = messages[sourceIndex + 1]
      const isFailedTurn = nextMessage?.role === 'assistant' && Boolean(nextMessage.error)
      const editedMessage: ChatMessage = { ...source, parts: [textPart(text)] }

      // Editing rewinds the conversation to this prompt — same as restore — so
      // drop the abandoned timeline's todos/background rows (and kill the live
      // processes) before the re-run repopulates them.
      clearSessionTodos(sessionId)
      resetSessionBackground(sessionId)
      clearPreviewArtifacts(sessionId)

      clearNotifications()
      setMutableRef(busyRef, true)
      setBusy(true)
      setAwaitingResponse(true)
      updateSessionState(sessionId, state => ({
        ...state,
        busy: true,
        awaitingResponse: true,
        pendingBranchGroup: null,
        sawAssistantPayload: false,
        interrupted: false,
        messages: [...state.messages.slice(0, sourceIndex), editedMessage]
      }))

      const isStaleTargetError = (err: unknown) =>
        /no longer in session history|not in session history/i.test(err instanceof Error ? err.message : String(err))

      try {
        await submitRewindPrompt(
          sessionId,
          text,
          isFailedTurn ? undefined : visibleUserOrdinal(messages, sourceIndex),
          busyRef.current || $busy.get()
        )
      } catch (err) {
        let surfaced = err

        if (!isFailedTurn && isStaleTargetError(err)) {
          try {
            // Already interrupted on the first attempt — submit as a plain resend.
            await submitRewindPrompt(sessionId, text, undefined, false)

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
      const visibleIds = new Set(nextMessages.map(m => m.id))
      const sessionId = activeSessionIdRef.current

      if (!sessionId) {
        return
      }

      updateSessionState(sessionId, state => {
        let changed = false

        const messages = state.messages.map(message => {
          if (message.role !== 'assistant' || !message.branchGroupId) {
            return message
          }

          const hidden = !visibleIds.has(message.id)

          if (message.hidden === hidden) {
            return message
          }

          changed = true

          return { ...message, hidden }
        })

        return changed ? { ...state, messages } : state
      })
    },
    [activeSessionIdRef, updateSessionState]
  )

  return {
    cancelRun,
    editMessage,
    handleThreadMessagesChange,
    handoffSession,
    reloadFromMessage,
    restoreToMessage,
    steerPrompt,
    submitText,
    transcribeVoiceAudio
  }
}

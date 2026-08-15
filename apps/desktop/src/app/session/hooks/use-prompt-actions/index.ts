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
  patchMainComposerAttachmentOccurrence,
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
  $currentCwd,
  $messages,
  $terminalBackend,
  setActiveSessionId,
  setAwaitingResponse,
  setBusy,
  setMessages,
  setTurnStartedAt
} from '@/store/session'
import { clearSessionSubagents } from '@/store/subagents'
import { clearSessionTodos } from '@/store/todos'
import { setSessionDraftingTool } from '@/store/tool-drafting'

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
  survivorRowIdsFrom,
  type SurvivorUserRowIds,
  truncateSubmitParams
} from './rewind'
import { useSlashCommand } from './slash'
import { useSubmitPrompt } from './submit'
import {
  blobToDataUrl,
  delay,
  friendlyRemoteAttachError,
  type GatewayRequest,
  inlineErrorMessage,
  markSessionRecentlyInterrupted,
  readFileDataUrlForAttach,
  readImageForRemoteAttach,
  shouldInterruptBeforeRewind,
  type SubmitTextOptions,
  withSessionNotFoundResume
} from './utils'

interface HandoffResult {
  ok: boolean
  error?: string
}

const WINDOWS_ABSOLUTE_PATH_RE = /^(?:[A-Za-z]:[\\/]|\\\\)/
const POSIX_ABSOLUTE_PATH_RE = /^\/(?!\/)/

// Terminal backends whose execution environment has its own filesystem
// (docker/ssh/singularity/modal/...) cannot see the desktop's host paths —
// they must be crossed as bytes, like remote attachments. Mirrors the
// container_backend set in tools/terminal_tool.py::_get_env_config.
const CONTAINER_TERMINAL_BACKENDS = new Set(['docker', 'ssh', 'singularity', 'modal', 'daytona', 'vercel_sandbox'])

// `mode: local` means the gateway was launched locally, not necessarily that
// Electron and the gateway share a filesystem. Windows Desktop can front a
// WSL/Docker backend whose cwd is POSIX, so a Windows host path must cross the
// boundary as bytes just like a remote attachment. Container terminal backends
// (docker, ssh, ...) always need bytes: the sandbox has its own filesystem and
// the host path would dangle inside it (#76577).
function attachmentPathNeedsUpload(path: string, backendCwd?: null | string, terminalBackend?: string): boolean {
  if (CONTAINER_TERMINAL_BACKENDS.has((terminalBackend || '').trim().toLowerCase())) {
    return true
  }

  return WINDOWS_ABSOLUTE_PATH_RE.test(path.trim()) && POSIX_ABSOLUTE_PATH_RE.test(backendCwd?.trim() || '')
}

/**
 * Stage one file/image attachment into the session workspace and return the
 * attachment rewritten with the gateway-side ref. Attachments upload their
 * bytes for remote gateways and local cross-filesystem backends; otherwise the
 * gateway receives the shared local path. Throws on failure so callers can
 * surface an error. Shared by submit-time sync, the eager drop-time upload, and
 * the message-edit composer drop — keep them in lockstep.
 */
export async function uploadComposerAttachment(
  attachment: ComposerAttachment,
  opts: {
    backendCwd?: null | string
    remote: boolean
    requestGateway: GatewayRequest
    sessionId: string
    /** Durable id used to re-register after sleep/wake or a backend restart. */
    storedSessionId?: null | string
    /** Called when the attach recovered onto a fresh live id. */
    onSessionRecovered?: (sessionId: string) => void
    terminalBackend?: string
  }
): Promise<ComposerAttachment> {
  const { backendCwd, remote, requestGateway, storedSessionId, onSessionRecovered, terminalBackend } = opts
  const path = attachment.path ?? ''
  const label = attachment.label || pathLabel(path)
  const uploadBytes = remote || attachmentPathNeedsUpload(path, backendCwd, terminalBackend)

  // Read bytes/paths ONCE, outside the retry. Only the session-scoped RPC is
  // replayed on recovery — re-reading a multi-MB file to retry a dead session
  // id would double the disk/IPC cost of every recovered attach. For images,
  // the chip's previewUrl already holds the full file as a base64 data URL,
  // so passing it avoids re-reading the same bytes off disk at submit.
  let imagePayload: Awaited<ReturnType<typeof readImageForRemoteAttach>> | null = null
  let fileDataUrl: null | string = null

  if (uploadBytes) {
    try {
      if (attachment.kind === 'image') {
        imagePayload = await readImageForRemoteAttach(path, attachment.previewUrl)
      } else {
        fileDataUrl = await readFileDataUrlForAttach(path)
      }
    } catch (err) {
      throw friendlyRemoteAttachError(err, label)
    }

    if (attachment.kind === 'image' ? !imagePayload : !fileDataUrl) {
      throw new Error(`Could not read ${label}`)
    }
  }

  const stageForSession = async (liveSessionId: string): Promise<ComposerAttachment> => {
    if (attachment.kind === 'image') {
      const result = imagePayload
        ? await requestGateway<ImageAttachResponse>('image.attach_bytes', {
            session_id: liveSessionId,
            content_base64: imagePayload.contentBase64,
            filename: imagePayload.filename
          })
        : await requestGateway<ImageAttachResponse>('image.attach', {
            path,
            session_id: liveSessionId
          })

      if (!result.attached) {
        throw new Error(result.message || `Could not attach ${label}`)
      }

      const attachedPath = result.path || path

      return {
        ...attachment,
        attachedSessionId: liveSessionId,
        label: attachedPath ? pathLabel(attachedPath) : attachment.label,
        path: attachedPath,
        uploadState: undefined
      }
    }

    const result = await requestGateway<FileAttachResponse>('file.attach', {
      name: label,
      path,
      session_id: liveSessionId,
      ...(fileDataUrl ? { data_url: fileDataUrl } : {})
    })

    if (!result.attached || !result.ref_text) {
      throw new Error(result.message || `Could not attach ${label}`)
    }

    return {
      ...attachment,
      attachedSessionId: liveSessionId,
      refText: result.ref_text,
      uploadState: undefined
    }
  }

  // Attach runs BEFORE prompt.submit, so submit's own recovery never gets a
  // chance: a stale runtime id fails here first and the user sees "session not
  // found" on an image while plain text works.
  const { result, sessionId: usedSessionId } = await withSessionNotFoundResume(
    opts.sessionId,
    storedSessionId,
    stageForSession,
    { requestGateway }
  )

  if (usedSessionId !== opts.sessionId) {
    onSessionRecovered?.(usedSessionId)
  }

  return result
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
  runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>>
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
  runtimeIdByStoredSessionIdRef,
  selectedStoredSessionIdRef,
  startFreshSessionDraft,
  sttEnabled,
  updateSessionState
}: PromptActionsOptions) {
  const { t } = useI18n()
  const copy = t.desktop

  const appendSessionTextMessage = useCallback(
    (
      sessionId: string,
      role: ChatMessage['role'],
      text: string,
      storedSessionId?: string | null,
      options: { appendAfterActiveReply?: boolean } = {}
    ) => {
      // Strip ANSI: slash-command output from the backend worker carries SGR
      // color codes (e.g. "Unknown command" in red). The ESC byte is invisible
      // in the chat panel, so without this the `[1;31m…[0m` payload leaks as
      // literal text.
      const body = stripAnsi(text).trim()

      if (!body) {
        return
      }

      const messageId = `${role}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`

      updateSessionState(
        sessionId,
        state => {
          const message: ChatMessage = {
            id: messageId,
            role,
            parts: [textPart(body)]
          }

          // Mid-turn correction: arrival order. The bubble lands after the
          // assistant output the user had already seen (sealing the live
          // stream so post-redirect deltas continue BELOW the correction),
          // never spliced above it (#73793) or mid-thread via the old
          // last-assistant fallback (#83151).
          if (options.appendAfterActiveReply) {
            return appendMidTurnUserMessage(state, message)
          }

          return { ...state, messages: [...state.messages, message] }
        },
        storedSessionId ?? selectedStoredSessionIdRef.current
      )

      return messageId
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
    ): Promise<{ attachments: ComposerAttachment[]; sessionId: string }> => {
      const updateComposerAttachments = options.updateComposerAttachments ?? true
      const remote = $connection.get()?.mode === 'remote'
      const storedSessionId = selectedStoredSessionIdRef.current
      let liveSessionId = sessionId
      const synced: ComposerAttachment[] = []

      const onSessionRecovered = (recoveredId: string) => {
        liveSessionId = recoveredId
        activeSessionIdRef.current = recoveredId
        setActiveSessionId(recoveredId)
      }

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
        // attachedSessionId) — don't re-upload it. Compare against the LIVE id:
        // after a mid-loop recovery an earlier chip's attachedSessionId points
        // at the dead runtime and must be re-staged.
        if (!attachment.path || attachment.attachedSessionId === liveSessionId) {
          synced.push(attachment)

          continue
        }

        if (attachment.kind === 'image' || attachment.kind === 'file') {
          const nextAttachment = await uploadComposerAttachment(attachment, {
            backendCwd: $currentCwd.get(),
            remote,
            requestGateway,
            sessionId: liveSessionId,
            storedSessionId,
            onSessionRecovered,
            terminalBackend: $terminalBackend.get()
          })

          // Update-only: never resurrect a chip the user removed mid-upload.
          if (updateComposerAttachments) {
            if (original.occurrenceId) {
              // Preserve a preview that may have completed after staging began,
              // while still refusing a remove + re-add of the same path.
              patchMainComposerAttachmentOccurrence(original, {
                attachedSessionId: nextAttachment.attachedSessionId,
                label: nextAttachment.label,
                path: nextAttachment.path,
                refText: nextAttachment.refText,
                uploadState: nextAttachment.uploadState
              })
            } else {
              updateComposerAttachment(nextAttachment)
            }
          }

          synced.push(nextAttachment)

          continue
        }

        synced.push(attachment)
      }

      return { attachments: synced, sessionId: liveSessionId }
    },
    [activeSessionIdRef, requestGateway, selectedStoredSessionIdRef]
  )

  // Stage a freshly dropped file as soon as it lands (when a session already
  // exists), so the upload runs while the user is still typing rather than
  // stalling the send. The card shows a spinner via `uploadState`; on success
  // the chip carries its gateway-side ref so submit skips re-uploading.
  //
  // Images are intentionally NOT eager-uploaded: thumbnail preparation already
  // performs a bounded full-file read/decode, and an eager upload would perform
  // another full-file read/IPC transfer immediately for every image in a batch.
  // Original bytes are still uploaded sequentially at submit via
  // image.attach_bytes.
  const eagerlyUploadAttachment = useCallback(
    async (sessionId: string, attachment: ComposerAttachment) => {
      const remote = $connection.get()?.mode === 'remote'

      setComposerAttachmentUploadState(attachment.id, 'uploading')

      try {
        // Update-only: if the user removed the chip while this was uploading,
        // don't resurrect it — just drop the staged result on the floor.
        updateComposerAttachment(
          await uploadComposerAttachment(attachment, {
            backendCwd: $currentCwd.get(),
            remote,
            requestGateway,
            sessionId,
            terminalBackend: $terminalBackend.get()
          })
        )
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
    runtimeIdByStoredSessionIdRef,
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
    getRoutedStoredSessionId,
    getRuntimeIdForStoredSession,
    handleSkinCommand,
    handoffSession,
    openMemoryGraph,
    refreshSessions,
    requestGateway,
    resumeStoredSession,
    selectedStoredSessionIdRef,
    startFreshSessionDraft,
    submitPromptText,
    updateSessionState
  })

  const submitText = useCallback(
    async (rawText: string, options?: SubmitTextOptions) => {
      const visibleText = sanitizeComposerInput(rawText).trim()
      const attachments = options?.attachments ?? $composerAttachments.get()

      if (!attachments.length && SLASH_COMMAND_RE.test(visibleText)) {
        triggerHaptic('selection')
        // Forward the explicit target (background queue drain, tile) — dropping
        // it ran the command against whatever chat happened to be in front.
        await executeSlashCommand(visibleText, options?.sessionId ? { sessionId: options.sessionId } : undefined)

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

    // Frontend busy clears immediately; gateway wind-down can lag. Mark so a
    // fast edit/resend still interrupt-first instead of racing 4009 (#83855).
    markSessionRecentlyInterrupted(sessionId)

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
        turnStartedAt: null,
        turnLive: false
      }
    })

    clearSessionTodos(sessionId)
    clearSessionSubagents(sessionId)
    resetSessionBackground(sessionId)
    setSessionDraftingTool(sessionId, '')
    // Stop ends the turn, so the gateway is no longer blocked on any prompt it
    // raised. Drop this session's pending clarify / approval / sudo / secret so
    // a dead panel (and the sidebar "needs input" dot) can't linger and accept
    // an answer the backend will reject.
    clearAllPrompts(sessionId)
    clearClarifyRequest(undefined, sessionId)

    try {
      await withSessionNotFoundResume(
        sessionId,
        selectedStoredSessionIdRef.current,
        liveId => requestGateway('session.interrupt', { session_id: liveId }),
        {
          requestGateway,
          onRecovered: recoveredId => {
            activeSessionIdRef.current = recoveredId
            setActiveSessionId(recoveredId)
          }
        }
      )
      releaseBusy()
    } catch (err) {
      releaseBusy()
      notifyError(err, copy.stopFailed)
    }
  }, [activeSessionIdRef, busyRef, copy.stopFailed, requestGateway, selectedStoredSessionIdRef, updateSessionState])

  // The desktop steering action is an immediate correction: the core cancels
  // model generation and rebuilds the live turn with displayed reasoning and
  // completed work intact. During a tool it waits for the safe result boundary.
  // Returns false when the turn raced to completion so the composer can queue.
  const redirectPrompt = useCallback(
    async (rawText: string): Promise<boolean> => {
      const text = sanitizeComposerInput(rawText).trim()
      // Ref, not the closure-captured prop — see cancelRun above. A redirect
      // reaches the live model mid-turn, so a stale target delivers the user's
      // correction into a conversation they are no longer looking at.
      const sessionId = activeSessionIdRef.current

      if (!text || !sessionId) {
        return false
      }

      // Accepted whether the live turn was redirected in place or queued for
      // the next turn (the build window, before the agent is wired) — either
      // way the correction reaches the model, so record it once as a real user
      // message after the interrupted checkpoint, matching the durable core
      // transcript rather than a system note that changes role after reload.
      const send = async (id: string): Promise<boolean> => {
        // Redirect aborts the model request, so the completion event can race
        // its RPC response. Record the correction *before* awaiting the
        // gateway, in arrival order: sealed already-streamed output above,
        // correction bubble below it, post-redirect deltas below that
        // (#73793, #83151).
        const messageId = appendSessionTextMessage(id, 'user', text, undefined, { appendAfterActiveReply: true })

        const discardOptimisticMessage = () =>
          updateSessionState(id, state => ({
            ...state,
            messages: state.messages.filter(message => message.id !== messageId)
          }))

        const moveOptimisticMessageToEnd = () =>
          updateSessionState(id, state => {
            const message = state.messages.find(candidate => candidate.id === messageId)

            return message
              ? { ...state, messages: [...state.messages.filter(candidate => candidate.id !== messageId), message] }
              : state
          })

        try {
          const result = await requestGateway<SessionRedirectResponse>('session.redirect', { session_id: id, text })

          if (result?.status === 'redirected') {
            triggerHaptic('submit')

            return true
          }

          if (result?.status === 'queued') {
            // Build-window redirects become the next turn, not part of the
            // active reply, so retain the optimistic row at the tail.
            moveOptimisticMessageToEnd()
            triggerHaptic('submit')

            return true
          }
        } catch (err) {
          discardOptimisticMessage()
          throw err
        }

        discardOptimisticMessage()

        return false
      }

      try {
        // A stale runtime id after reconnect 404s ("session not found"): the
        // shared resolver resumes the stored session and retries once, so a
        // correction right after a reconnect isn't lost to the race.
        const { result } = await withSessionNotFoundResume(sessionId, selectedStoredSessionIdRef.current, send, {
          requestGateway,
          onRecovered: recoveredId => {
            activeSessionIdRef.current = recoveredId
            setActiveSessionId(recoveredId)
          }
        })

        return result
      } catch {
        // Swallow — caller queues the text so nothing is lost.
      }

      return false
    },
    [activeSessionIdRef, appendSessionTextMessage, requestGateway, selectedStoredSessionIdRef, updateSessionState]
  )

  // After a durable rewind the surviving bubbles' cached rowIds are stale (the
  // gateway re-inserted the kept prefix as new SQLite rows). Rebind them to the
  // authoritative post-rewrite ids so the NEXT rewind/edit/regenerate doesn't
  // send a dead id and get refused with 4018 (consecutive-rewind staleness,
  // #83202 review).
  const applySurvivorRowIds = useCallback(
    (sessionId: string, survivorRowIds: SurvivorUserRowIds | undefined) => {
      if (!survivorRowIds) {
        return
      }

      updateSessionState(sessionId, state => ({
        ...state,
        messages: rebindSurvivorRowIds(state.messages, survivorRowIds)
      }))
    },
    [updateSessionState]
  )

  const reloadFromMessage = useCallback(
    async (parentId: string | null) => {
      // Ref, not the closure-captured prop — a truncating resubmit aimed at a
      // stale session deletes the wrong transcript.
      const sessionId = activeSessionIdRef.current

      if (!sessionId || $busy.get()) {
        return
      }

      const plan = planReload($messages.get(), parentId)

      if (!plan) {
        return
      }

      clearNotifications()
      updateSessionState(sessionId, state => applyReloadOptimistic(state, plan))

      try {
        const result = await requestGateway<{ survivor_user_row_ids?: unknown }>(
          'prompt.submit',
          {
            session_id: sessionId,
            text: plan.text,
            ...truncateSubmitParams(plan.truncateOrdinal, plan.truncateMessageId, plan.truncateRowId)
          },
          PROMPT_SUBMIT_REQUEST_TIMEOUT_MS
        )

        applySurvivorRowIds(sessionId, survivorRowIdsFrom(result))
      } catch (err) {
        updateSessionState(sessionId, state => ({
          ...state,
          busy: false,
          awaitingResponse: false
        }))
        notifyError(err, copy.regenerateFailed)
      }
    },
    [activeSessionIdRef, applySurvivorRowIds, copy.regenerateFailed, requestGateway, updateSessionState]
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
    (
      sessionId: string,
      text: string,
      truncateOrdinal: number | undefined,
      truncateMessageId: string | undefined,
      interruptFirst: boolean,
      truncateRowId?: number
    ) =>
      runRewindSubmit(
        requestGateway,
        sessionId,
        text,
        truncateOrdinal,
        truncateMessageId,
        interruptFirst,
        {
          storedSessionId: selectedStoredSessionIdRef.current,
          onSessionRecovered: recoveredId => {
            activeSessionIdRef.current = recoveredId
            setActiveSessionId(recoveredId)
          }
        },
        truncateRowId
      ),
    [activeSessionIdRef, requestGateway, selectedStoredSessionIdRef]
  )

  const restoreToMessage = useCallback(
    async (messageId: string, target?: RestoreMessageTarget) => {
      // Ref, not the closure-captured prop — a rewind is destructive, so a
      // stale target truncates a conversation the user did not ask to rewind.
      const sessionId = activeSessionIdRef.current

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

      // Capture before optimistic busy=true — otherwise interruptFirst is always
      // true and idle restores wrongly interrupt (and Stop→edit misses cooldown).
      const interruptFirst = shouldInterruptBeforeRewind({
        busy: busyRef.current || $busy.get(),
        sessionId
      })

      clearNotifications()
      setMutableRef(busyRef, true)
      setBusy(true)
      setAwaitingResponse(true)
      updateSessionState(sessionId, state => applyRewindOptimistic(state, plan.sourceIndex))

      try {
        const survivorRowIds = await submitRewindPrompt(
          sessionId,
          plan.text,
          plan.truncateOrdinal,
          plan.truncateMessageId,
          interruptFirst,
          plan.truncateRowId
        )

        applySurvivorRowIds(sessionId, survivorRowIds)
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
    [activeSessionIdRef, applySurvivorRowIds, busyRef, submitRewindPrompt, updateSessionState]
  )

  const editMessage = useCallback(
    async (edited: AppendMessage) => {
      // Ref, not the closure-captured prop — an edit rewinds and resubmits, so
      // a stale target rewrites the wrong session's history.
      const sessionId = activeSessionIdRef.current
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

      // Before optimistic busy=true — see restoreToMessage (#83855).
      const interruptFirst = shouldInterruptBeforeRewind({
        busy: busyRef.current || $busy.get(),
        sessionId
      })

      clearNotifications()
      setMutableRef(busyRef, true)
      setBusy(true)
      setAwaitingResponse(true)
      updateSessionState(sessionId, state => applyRewindOptimistic(state, plan.sourceIndex, plan.editedMessage))

      try {
        const survivorRowIds = await submitRewindPrompt(
          sessionId,
          plan.text,
          plan.truncateOrdinal,
          plan.truncateMessageId,
          interruptFirst,
          plan.truncateRowId
        )

        applySurvivorRowIds(sessionId, survivorRowIds)
      } catch (err) {
        // Roll the optimistic edit/truncation back to the original history so the
        // UI stays in sync with what's persisted instead of stranding a partial
        // timeline.
        setMutableRef(busyRef, false)
        setBusy(false)
        setAwaitingResponse(false)
        updateSessionState(sessionId, state => ({ ...state, busy: false, awaitingResponse: false, messages }))
        notifyError(err, copy.editFailed)
      }
    },
    [activeSessionIdRef, applySurvivorRowIds, busyRef, copy.editFailed, submitRewindPrompt, updateSessionState]
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

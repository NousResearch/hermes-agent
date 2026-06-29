import type { AppendMessage, ThreadMessage } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { type MutableRefObject, useCallback, useEffect, useRef } from 'react'

import { getProfiles, transcribeAudio } from '@/hermes'
import { translateNow, type Translations, useI18n } from '@/i18n'
import { stripAnsi } from '@/lib/ansi'
import { branchGroupForUser, type ChatMessage, chatMessageText, textPart } from '@/lib/chat-messages'
import {
  optimisticAttachmentRef,
  parseCommandDispatch,
  parseSlashCommand,
  pathLabel,
  sessionTitle,
  SLASH_COMMAND_RE
} from '@/lib/chat-runtime'
import {
  type CommandsCatalogLike,
  type DesktopActionId,
  type DesktopPickerId,
  desktopSlashUnavailableMessage,
  filterDesktopCommandsCatalog,
  isDesktopSlashCommand,
  resolveDesktopCommand
} from '@/lib/desktop-slash-commands'
import { triggerHaptic } from '@/lib/haptics'
import { setMutableRef } from '@/lib/mutable-ref'
import { isProviderSetupErrorMessage } from '@/lib/provider-setup-errors'
import { setSessionYolo } from '@/lib/yolo-session'
import { clearClarifyRequest } from '@/store/clarify'
import { openCommandPalettePage } from '@/store/command-palette'
import {
  $composerAttachments,
  clearComposerAttachments,
  type ComposerAttachment,
  setComposerAttachmentUploadState,
  setComposerDraft,
  terminalContextBlocksFromDraft,
  updateComposerAttachment
} from '@/store/composer'
import { resetSessionBackground } from '@/store/composer-status'
import { clearNotifications, notify, notifyError } from '@/store/notifications'
import { requestDesktopOnboarding } from '@/store/onboarding'
import { setPetScale } from '@/store/pet-gallery'
import { $petGenInput, openPetGenerate } from '@/store/pet-generate'
import { clearPreviewArtifacts } from '@/store/preview-status'
import { $activeGatewayProfile, $newChatProfile, ensureGatewayProfile, normalizeProfileKey } from '@/store/profile'
import { clearAllPrompts } from '@/store/prompts'
import {
  $busy,
  $connection,
  $messages,
  $sessions,
  $yoloActive,
  setAwaitingResponse,
  setBusy,
  setMessages,
  setModelPickerOpen,
  setSessionPickerOpen,
  setSessions,
  setYoloActive
} from '@/store/session'
import { clearSessionSubagents } from '@/store/subagents'
import { clearSessionTodos } from '@/store/todos'

import type {
  BrowserManageResponse,
  ClientSessionState,
  FileAttachResponse,
  HandoffFailResponse,
  HandoffRequestResponse,
  HandoffStateResponse,
  ImageAttachResponse,
  SessionSteerResponse,
  SessionTitleResponse,
  SlashExecResponse
} from '../../types'

interface HandoffResult {
  ok: boolean
  error?: string
}

function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

function isSessionIdCandidate(value: string): boolean {
  const trimmed = value.trim()

  return /^\d{8}_\d{6}_[A-Fa-f0-9]{6}$/.test(trimmed) || /^[A-Fa-f0-9]{32}$/.test(trimmed)
}

function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()

    reader.addEventListener('load', () => {
      if (typeof reader.result === 'string') {
        resolve(reader.result)
      } else {
        reject(new Error(translateNow('desktop.audioReadFailed')))
      }
    })
    reader.addEventListener('error', () => reject(reader.error || new Error(translateNow('desktop.audioReadFailed'))))
    reader.readAsDataURL(blob)
  })
}

function isProviderSetupError(error: unknown) {
  const message = error instanceof Error ? error.message : String(error)

  return isProviderSetupErrorMessage(message)
}

function inlineErrorMessage(error: unknown, fallback: string): string {
  const raw = error instanceof Error ? error.message : typeof error === 'string' ? error : fallback

  return (raw.match(/Error invoking remote method '[^']+': Error: (.+)$/)?.[1] ?? raw).replace(/^Error:\s*/, '').trim()
}

function isPromptSubmitTimeoutError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error)

  return /request timed out:\s*prompt\.submit/i.test(message)
}

function submitErrorMessage(error: unknown, copy: Translations['desktop']): string {
  if (isPromptSubmitTimeoutError(error)) {
    return copy.promptSubmitTimedOut
  }

  return inlineErrorMessage(error, copy.promptFailed)
}

function isSessionNotFoundError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error)

  return /session not found/i.test(message)
}

// The gateway refuses prompt.submit while a turn is running (4009 "session
// busy"). It's a transient concurrency guard, never a user-facing error: a
// submit racing the settle edge (or a rewind interrupting mid-turn) just waits
// a beat for the turn to wind down, then lands. Bounded so a genuinely stuck
// turn still surfaces eventually.
const SESSION_BUSY_RETRY_TIMEOUT_MS = 6_000
const SESSION_BUSY_RETRY_INTERVAL_MS = 150

function isSessionBusyError(error: unknown): boolean {
  return /session busy/i.test(error instanceof Error ? error.message : String(error))
}

const sleep = (ms: number) => new Promise<void>(resolve => setTimeout(resolve, ms))

// Retry a gateway call across transient "session busy" so it never reaches the
// user — the turn settles within the deadline and the call lands.
async function withSessionBusyRetry<T>(call: () => Promise<T>): Promise<T> {
  const deadline = Date.now() + SESSION_BUSY_RETRY_TIMEOUT_MS

  for (;;) {
    try {
      return await call()
    } catch (err) {
      if (isSessionBusyError(err) && Date.now() < deadline) {
        await sleep(SESSION_BUSY_RETRY_INTERVAL_MS)

        continue
      }

      throw err
    }
  }
}

// Hard guard: at most one prompt.submit in flight per session. Every submit
// path — user Enter, queue drain, busy-retry, slash fallthrough — funnels
// through submitPromptText. Without this, a stalled turn (e.g. a context-bloated
// session whose first call hangs) let the SAME prompt launch several real turns
// at once (the "message stacked 5×" bug). Keyed by stored/active session id.
const _submitInFlight = new Set<string>()

function base64FromDataUrl(dataUrl: string): string {
  const comma = dataUrl.indexOf(',')

  return comma >= 0 ? dataUrl.slice(comma + 1) : ''
}

function imageFilenameFromPath(filePath: string): string {
  return filePath.split(/[\\/]/).filter(Boolean).pop() || 'image.png'
}

// Remote gateway: the local composer-image file lives on THIS machine's disk,
// not the gateway's, so read the bytes here and upload them via
// image.attach_bytes. Returns null when the file can't be read.
async function readImageForRemoteAttach(filePath: string): Promise<{ contentBase64: string; filename: string } | null> {
  const dataUrl = await window.hermesDesktop?.readFileDataUrl(filePath)
  const contentBase64 = dataUrl ? base64FromDataUrl(dataUrl) : ''

  return contentBase64 ? { contentBase64, filename: imageFilenameFromPath(filePath) } : null
}

// Read a non-image file as a data URL for upload via file.attach. Returns null
// when the desktop bridge can't read the file (e.g. it was moved/deleted).
async function readFileDataUrlForAttach(filePath: string): Promise<string | null> {
  const reader = window.hermesDesktop?.readFileDataUrl

  if (!reader) {
    return null
  }

  const dataUrl = await reader(filePath)

  return dataUrl || null
}

// The readFileDataUrl IPC base64-loads the whole file into memory and is
// hard-capped (DATA_URL_READ_MAX_BYTES, 16 MB) in electron/hardening.cjs, which
// rejects with a raw "file is too large (N bytes; limit M bytes)" string. In
// remote mode every attachment's bytes go through that read, so a big file
// surfaces that internal message verbatim in the failure toast. Translate it
// into a friendly "too large to upload to the remote gateway" line, parsing the
// limit out of the message so it tracks the real cap. Non-cap errors pass
// through unchanged.
function friendlyRemoteAttachError(err: unknown, label: string): Error {
  const message = err instanceof Error ? err.message : String(err)

  if (!/too large/i.test(message)) {
    return err instanceof Error ? err : new Error(message)
  }

  const limitBytes = Number(message.match(/limit (\d+) bytes/)?.[1])
  const cap = Number.isFinite(limitBytes) && limitBytes > 0 ? ` (max ${Math.floor(limitBytes / (1024 * 1024))} MB)` : ''

  return new Error(`${label} is too large to upload to the remote gateway${cap}.`)
}

type GatewayRequest = <T>(method: string, params?: Record<string, unknown>) => Promise<T>

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
  handleSkinCommand: (arg: string) => string
  refreshSessions: () => Promise<void>
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
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

interface SubmitTextOptions {
  attachments?: ComposerAttachment[]
  fromQueue?: boolean
}

/** Everything a slash handler needs about the invocation it's serving. */
interface SlashActionCtx {
  arg: string
  command: string
  name: string
  recordInput: boolean
  sessionHint?: string
}

function renderCommandsCatalog(catalog: CommandsCatalogLike, copy: Translations['desktop']): string {
  const desktopCatalog = filterDesktopCommandsCatalog(catalog)

  const sections = desktopCatalog.categories?.length
    ? desktopCatalog.categories
    : [{ name: copy.desktopCommands, pairs: desktopCatalog.pairs ?? [] }]

  const body = sections
    .filter(section => section.pairs.length > 0)
    .map(section => {
      const rows = section.pairs.map(([cmd, desc]) => `${cmd.padEnd(18)} ${desc}`)

      return [`${section.name}:`, ...rows].join('\n')
    })
    .join('\n\n')

  const tail = [
    desktopCatalog.skill_count ? copy.skillCommandsAvailable(desktopCatalog.skill_count) : '',
    desktopCatalog.warning ? copy.warningLine(desktopCatalog.warning) : ''
  ]
    .filter(Boolean)
    .join('\n')

  return [body || 'No desktop commands available.', tail].filter(Boolean).join('\n\n')
}

function slashStatusText(command: string, output: string): string {
  return [`slash:${command}`, output.trim()].filter(Boolean).join('\n')
}

function appendText(message: AppendMessage): string {
  return message.content
    .map(part => ('text' in part ? part.text : ''))
    .join('')
    .trim()
}

function visibleUserOrdinal(messages: readonly ChatMessage[], end: number): number {
  return messages.slice(0, end).filter(m => m.role === 'user' && !m.hidden).length
}

function visibleUserIndexAtOrdinal(messages: readonly ChatMessage[], targetOrdinal: number): number {
  let ordinal = 0

  for (let index = 0; index < messages.length; index += 1) {
    const message = messages[index]

    if (message.role !== 'user' || message.hidden) {
      continue
    }

    if (ordinal === targetOrdinal) {
      return index
    }

    ordinal += 1
  }

  return -1
}

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
  handleSkinCommand,
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

  const submitPromptText = useCallback(
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

      // One submit in flight per session — drop any concurrent re-fire so a
      // stalled turn can't stack the same prompt into multiple real turns.
      const submitLockKey = selectedStoredSessionIdRef.current || activeSessionId || '__pending_new__'

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
          selectedStoredSessionIdRef.current
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
          selectedStoredSessionIdRef.current
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
          selectedStoredSessionIdRef.current
        )
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
          dropOptimistic(null)
          releaseBusy()
          notify({ kind: 'error', title: copy.sessionUnavailable, message: copy.createSessionFailed })

          return false
        }

        seedOptimistic(sessionId)
      }

      try {
        const syncedAttachments = await syncAttachmentsForSubmit(sessionId, attachments, {
          updateComposerAttachments: usingComposerAttachments
        })

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
          await withSessionBusyRetry(() => requestGateway('prompt.submit', { session_id: sessionId, text }))
        } catch (firstErr) {
          if (isSessionNotFoundError(firstErr) && selectedStoredSessionIdRef.current) {
            // Re-register the session in the gateway and get a fresh live ID.
            const resumed = await requestGateway<{ session_id: string }>('session.resume', {
              session_id: selectedStoredSessionIdRef.current
            })

            const recoveredId = resumed?.session_id

            if (recoveredId) {
              activeSessionIdRef.current = recoveredId
              await withSessionBusyRetry(() => requestGateway('prompt.submit', { session_id: recoveredId, text }))
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

        const message = submitErrorMessage(err, copy)

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

        if (isPromptSubmitTimeoutError(err)) {
          notify({ kind: 'error', title: copy.promptFailed, message })

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
      requestGateway,
      selectedStoredSessionIdRef,
      syncAttachmentsForSubmit,
      updateSessionState
    ]
  )

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

      const target = platform.trim().toLowerCase()

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

  const executeSlashCommand = useCallback(
    async (rawCommand: string, options?: { sessionId?: string; recordInput?: boolean }) => {
      const ensureSessionId = async (sessionHint?: string) =>
        sessionHint || activeSessionIdRef.current || (await createBackendSessionForSend())

      // Resolve the target session plus a writer for inline slash output, or
      // notify + return null when none can be created. Folds the ensure / bail /
      // build-renderSlashOutput boilerplate every exec-style handler repeats.
      const withSlashOutput = async (
        ctx: SlashActionCtx
      ): Promise<{ render: (text: string) => void; sessionId: string } | null> => {
        const sessionId = await ensureSessionId(ctx.sessionHint)

        if (!sessionId) {
          notify({ kind: 'error', title: copy.sessionUnavailable, message: copy.createSessionFailed })

          return null
        }

        const render = (text: string) =>
          appendSessionTextMessage(sessionId, 'system', ctx.recordInput ? slashStatusText(ctx.command, text) : text)

        return { render, sessionId }
      }

      // `exec` commands (and unknown skill / quick commands the backend owns)
      // run on the gateway and render their text output inline. This is the only
      // path that talks to slash.exec / command.dispatch.
      async function runExec(ctx: SlashActionCtx): Promise<void> {
        const { arg, command, name } = ctx
        const resolved = await withSlashOutput(ctx)

        if (!resolved) {
          return
        }

        const { render: renderSlashOutput, sessionId } = resolved

        if (!isDesktopSlashCommand(name)) {
          renderSlashOutput(desktopSlashUnavailableMessage(name) || `/${name} is not available in the desktop app.`)

          return
        }

        const handleDispatch = async (
          dispatch: NonNullable<ReturnType<typeof parseCommandDispatch>>
        ): Promise<void> => {
          if (dispatch.type === 'exec' || dispatch.type === 'plugin') {
            renderSlashOutput(dispatch.output ?? '(no output)')

            return
          }

          if (dispatch.type === 'alias') {
            await runSlash(`/${dispatch.target}${arg ? ` ${arg}` : ''}`, sessionId, false)

            return
          }

          // send / prefill carry an optional `notice` (e.g. "⊙ Goal set …")
          // that the backend wants shown as a system line before the message
          // is acted on. Mirrors the TUI's createSlashHandler — without it a
          // `/goal <text>` looked like it did nothing.
          if ((dispatch.type === 'send' || dispatch.type === 'prefill') && dispatch.notice?.trim()) {
            renderSlashOutput(dispatch.notice.trim())
          }

          const message = ('message' in dispatch ? dispatch.message : '')?.trim() ?? ''

          // /undo returns a prefill directive: drop the backed-up message into
          // the composer for editing instead of submitting it immediately.
          if (dispatch.type === 'prefill') {
            if (message) {
              setComposerDraft(message)
            }

            return
          }

          if (!message) {
            renderSlashOutput(
              `/${name}: ${dispatch.type === 'skill' ? 'skill payload missing message' : 'empty message'}`
            )

            return
          }

          if (dispatch.type === 'skill') {
            renderSlashOutput(`⚡ loading skill: ${dispatch.name}`)
          }

          if (busyRef.current) {
            renderSlashOutput('session busy — /interrupt the current turn before sending this command')

            return
          }

          await submitPromptText(message)
        }

        try {
          const result = await requestGateway<unknown>('slash.exec', {
            session_id: sessionId,
            command: command.replace(/^\/+/, '')
          })

          const dispatch = parseCommandDispatch(result)

          if (dispatch) {
            await handleDispatch(dispatch)

            return
          }

          const output = result && typeof result === 'object' ? (result as SlashExecResponse) : null
          const body = output?.output || `/${name}: no output`
          renderSlashOutput(output?.warning ? `warning: ${output.warning}\n${body}` : body)

          return
        } catch {
          // Fall back to command.dispatch for skill/send/alias directives.
        }

        try {
          const dispatch = parseCommandDispatch(
            await requestGateway<unknown>('command.dispatch', { session_id: sessionId, name, arg })
          )

          if (!dispatch) {
            renderSlashOutput('error: invalid response: command.dispatch')

            return
          }

          await handleDispatch(dispatch)
        } catch (err) {
          renderSlashOutput(`error: ${err instanceof Error ? err.message : String(err)}`)
        }
      }

      // One handler per `action` command. Adding a desktop-native command is a
      // registry row in desktop-slash-commands.ts plus an entry here — never a
      // new branch in a dispatch ladder.
      const actionHandlers: Record<DesktopActionId, (ctx: SlashActionCtx) => Promise<void>> = {
        new: async () => {
          startFreshSessionDraft()
        },
        branch: async () => {
          await branchCurrentSession()
        },
        // /yolo maps to the status-bar YOLO control — a per-session approval
        // bypass, same scope as the TUI's Shift+Tab. With no session yet we arm
        // it locally; the session-create path applies it on the first message.
        yolo: async ({ sessionHint }) => {
          const sid = sessionHint || activeSessionIdRef.current
          const next = !$yoloActive.get()

          if (!sid) {
            setYoloActive(next)
            notify({ kind: 'success', message: next ? copy.yoloArmed : copy.yoloOff })

            return
          }

          try {
            const active = await setSessionYolo(requestGateway, sid, next)
            appendSessionTextMessage(sid, 'system', copy.yoloSystem(active))
          } catch {
            notify({ kind: 'error', title: copy.yoloTitle, message: copy.yoloToggleFailed })
          }
        },
        // /handoff hands this session to a messaging platform. The platform is
        // completed inline in the slash popover (backend _handoff_completions),
        // so there is no overlay: `/handoff <platform>` runs the desktop's own
        // handoff RPC. cli_only on the backend, so it must not reach slash.exec.
        handoff: async ({ arg, command, recordInput, sessionHint }) => {
          const platform = arg.trim()

          if (!platform) {
            notify({ kind: 'success', message: copy.handoff.pickPlatform })

            return
          }

          const sid = sessionHint || activeSessionIdRef.current

          if (!sid) {
            notify({ kind: 'error', title: copy.sessionUnavailable, message: copy.createSessionFailed })

            return
          }

          const result = await handoffSession(platform, { sessionId: sid })

          if (!result.ok && result.error) {
            appendSessionTextMessage(sid, 'system', recordInput ? slashStatusText(command, result.error) : result.error)
          }
        },
        // /profile selects which profile new chats open in — no app relaunch.
        // A profile is per-session now, so an existing thread can't change its
        // profile mid-stream; `/profile <name>` points the next new chat (and
        // the current empty draft) at that profile's backend.
        profile: async ({ arg }) => {
          const target = arg.trim()
          const current = normalizeProfileKey($activeGatewayProfile.get())

          if (!target) {
            notify({ kind: 'success', message: copy.profileStatus(current) })

            return
          }

          try {
            const { profiles } = await getProfiles()
            const match = profiles.find(profile => profile.name === target)

            if (!match) {
              notify({
                kind: 'error',
                title: copy.unknownProfile,
                message: copy.noProfileNamed(target, profiles.map(profile => profile.name).join(', '))
              })

              return
            }

            const key = normalizeProfileKey(match.name)

            $newChatProfile.set(key)
            await ensureGatewayProfile(key)
            notify({ kind: 'success', message: copy.newChatsProfile(match.name) })
          } catch (err) {
            notifyError(err, copy.setProfileFailed)
          }
        },
        skin: async ({ arg, command, recordInput, sessionHint }) => {
          const sid = sessionHint || activeSessionIdRef.current
          const message = handleSkinCommand(arg)

          // No session to print into yet — surface it as a toast instead of
          // spinning up a backend session just to change the theme.
          if (!sid) {
            notify({ kind: 'success', message })

            return
          }

          appendSessionTextMessage(sid, 'system', recordInput ? slashStatusText(command, message) : message)
        },
        // /title <name> renames via the gateway's session.title RPC — the same
        // path the TUI uses, NOT REST renameSession (which 404s on runtime ids)
        // nor the slash worker (whose DB write can silently fail). Bare /title
        // shows the current title, which the worker owns, so delegate to exec.
        title: async ctx => {
          if (!ctx.arg) {
            await runExec(ctx)

            return
          }

          const resolved = await withSlashOutput(ctx)

          if (!resolved) {
            return
          }

          const { render: renderSlashOutput, sessionId } = resolved
          const { arg } = ctx

          try {
            const result = await requestGateway<SessionTitleResponse>('session.title', {
              session_id: sessionId,
              title: arg
            })

            const finalTitle = (result?.title || arg).trim()
            const queued = result?.pending === true

            setSessions(prev => prev.map(s => (s.id === sessionId ? { ...s, title: finalTitle || null } : s)))
            await refreshSessions().catch(() => undefined)
            renderSlashOutput(
              finalTitle
                ? `Session title set: ${finalTitle}${queued ? ' (queued while session initializes)' : ''}`
                : 'Session title cleared.'
            )
          } catch (err) {
            renderSlashOutput(`error: ${err instanceof Error ? err.message : String(err)}`)
          }
        },
        help: async ctx => {
          const resolved = await withSlashOutput(ctx)

          if (!resolved) {
            return
          }

          const { render: renderSlashOutput, sessionId } = resolved

          try {
            const catalog = await requestGateway<CommandsCatalogLike>('commands.catalog', { session_id: sessionId })

            renderSlashOutput(renderCommandsCatalog(catalog, copy))
          } catch (err) {
            renderSlashOutput(`error: ${err instanceof Error ? err.message : String(err)}`)
          }
        },
        // /hatch opens the pet generator overlay (the desktop's rich, multi-step
        // generate→pick→hatch→adopt flow). A typed description seeds the prompt
        // so `/hatch a cyber fox` lands on the composer step prefilled.
        hatch: async ({ arg }) => {
          const concept = arg.trim()

          if (concept) {
            $petGenInput.set(concept)
          }

          openPetGenerate()
        },
        pet: async ctx => {
          const [sub = '', rawValue = ''] = ctx.arg.trim().split(/\s+/)
          const lower = sub.toLowerCase()

          if (lower === 'list' || lower === 'gallery' || lower === 'browse' || lower === 'all') {
            openCommandPalettePage('pets')

            return
          }

          // `/pet scale <n>` resizes the floating pet locally (instant) and
          // persists via the store — no round-trip to the slash worker.
          if (lower === 'scale') {
            const value = Number(rawValue)

            if (!rawValue || Number.isNaN(value)) {
              const resolved = await withSlashOutput(ctx)
              resolved?.render('usage: /pet scale <factor>  (e.g. /pet scale 0.5)')

              return
            }

            setPetScale(requestGateway, value)

            return
          }

          await runExec(ctx)
        },
        // /browser connect|disconnect|status manages the live CDP connection on
        // the gateway host, mirroring the TUI's browser.manage RPC. It mutates
        // BROWSER_CDP_URL (and may launch Chrome) in the gateway process — only
        // meaningful when that process runs on this machine, so it's gated to
        // local connections. A remote gateway would act on the wrong host.
        browser: async ctx => {
          const resolved = await withSlashOutput(ctx)

          if (!resolved) {
            return
          }

          const { render: renderSlashOutput, sessionId } = resolved

          if ($connection.get()?.mode === 'remote') {
            renderSlashOutput(
              '/browser manages a Chromium-family browser on the gateway host — only available when connected to a local gateway.'
            )

            return
          }

          const [rawAction = 'status', ...rest] = ctx.arg.trim().split(/\s+/).filter(Boolean)
          const cmdAction = rawAction.toLowerCase()

          if (!['connect', 'disconnect', 'status'].includes(cmdAction)) {
            renderSlashOutput(
              'usage: /browser [connect|disconnect|status] [url] · persistent: set browser.cdp_url in config.yaml'
            )

            return
          }

          const url = cmdAction === 'connect' ? rest.join(' ').trim() || 'http://127.0.0.1:9222' : undefined

          if (url) {
            renderSlashOutput(`checking Chromium-family browser remote debugging at ${url}...`)
          }

          try {
            const result = await requestGateway<BrowserManageResponse>('browser.manage', {
              action: cmdAction,
              session_id: sessionId,
              ...(url && { url })
            })

            // Without a streamed session subscription, the gateway bundles its
            // progress lines into `messages` — flush them inline.
            result?.messages?.forEach(message => renderSlashOutput(message))

            if (cmdAction === 'status') {
              renderSlashOutput(
                result?.connected
                  ? `browser connected: ${result.url || '(url unavailable)'}`
                  : 'browser not connected (try /browser connect <url> or set browser.cdp_url in config.yaml)'
              )

              return
            }

            if (cmdAction === 'disconnect') {
              renderSlashOutput('browser disconnected')

              return
            }

            if (result?.connected) {
              renderSlashOutput('Browser connected to live Chromium-family browser via CDP')
              renderSlashOutput(`Endpoint: ${result.url || '(url unavailable)'}`)
              renderSlashOutput('next browser tool call will use this CDP endpoint')
            }
          } catch (err) {
            renderSlashOutput(`error: ${err instanceof Error ? err.message : String(err)}`)
          }
        }
      }

      // Picker commands open a desktop overlay; a typed arg is resolved by that
      // picker so the command never dead-ends or falls through to the backend.
      const openPicker = async (pickerId: DesktopPickerId, ctx: SlashActionCtx): Promise<void> => {
        if (pickerId === 'model') {
          if (!ctx.arg.trim()) {
            setModelPickerOpen(true)

            return
          }

          // Power users can still type `/model <name>` — run it on the backend.
          await runExec(ctx)

          return
        }

        // session picker — /resume, /sessions, /switch
        const query = ctx.arg.trim()

        if (!query) {
          setSessionPickerOpen(true)

          return
        }

        const sessions = $sessions.get()
        const lower = query.toLowerCase()

        const match =
          sessions.find(session => session.id === query) ||
          sessions.find(session => sessionTitle(session).toLowerCase().includes(lower)) ||
          sessions.find(session => (session.preview ?? '').toLowerCase().includes(lower))

        if (!match) {
          if (isSessionIdCandidate(query)) {
            await resumeStoredSession(query)

            return
          }

          notify({ kind: 'error', message: copy.resumeFailed })

          return
        }

        await resumeStoredSession(match.id)
      }

      // The whole dispatcher: resolve the command's desktop surface, then act on
      // its kind. No per-command ladder — behavior lives in the registry.
      async function runSlash(commandText: string, sessionHint?: string, recordInput = true): Promise<void> {
        const command = commandText.trim()
        const { name, arg } = parseSlashCommand(command)

        if (!name) {
          const sessionId = await ensureSessionId(sessionHint)

          if (sessionId) {
            appendSessionTextMessage(sessionId, 'system', copy.emptySlashCommand)
          }

          return
        }

        const ctx: SlashActionCtx = { arg, command, name, recordInput, sessionHint }
        const surface = resolveDesktopCommand(`/${name}`)?.surface

        switch (surface?.kind) {
          case 'unavailable': {
            const resolved = await withSlashOutput(ctx)
            resolved?.render(desktopSlashUnavailableMessage(name) || `/${name} is not available in the desktop app.`)

            return
          }

          case 'picker':
            return openPicker(surface.picker, ctx)

          case 'action':
            return actionHandlers[surface.action](ctx)

          default:
            // exec spec, or an unknown skill / quick command the backend owns.
            return runExec(ctx)
        }
      }

      await runSlash(rawCommand, options?.sessionId, options?.recordInput ?? true)
    },
    [
      activeSessionIdRef,
      appendSessionTextMessage,
      branchCurrentSession,
      busyRef,
      copy,
      createBackendSessionForSend,
      handleSkinCommand,
      handoffSession,
      refreshSessions,
      requestGateway,
      resumeStoredSession,
      startFreshSessionDraft,
      submitPromptText
    ]
  )

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
            session_id: selectedStoredSessionIdRef.current
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
        await requestGateway('prompt.submit', {
          session_id: activeSessionId,
          text: userText,
          truncate_before_user_ordinal: truncateBeforeUserOrdinal
        })
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
        requestGateway('prompt.submit', {
          session_id: sessionId,
          text,
          ...(truncateOrdinal !== undefined && { truncate_before_user_ordinal: truncateOrdinal })
        })

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

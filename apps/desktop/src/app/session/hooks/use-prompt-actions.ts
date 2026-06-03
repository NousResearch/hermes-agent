import type { AppendMessage, ThreadMessage } from '@assistant-ui/react'
import { type MutableRefObject, useCallback } from 'react'

import { transcribeAudio } from '@/hermes'
import { appendTextPart, branchGroupForUser, type ChatMessage, chatMessageText, textPart } from '@/lib/chat-messages'
import {
  attachmentDisplayText,
  INTERRUPTED_MARKER,
  parseCommandDispatch,
  parseSlashCommand,
  pathLabel,
  SLASH_COMMAND_RE
} from '@/lib/chat-runtime'
import {
  type CommandsCatalogLike,
  desktopSlashUnavailableMessage,
  filterDesktopCommandsCatalog,
  isDesktopSlashCommand
} from '@/lib/desktop-slash-commands'
import { triggerHaptic } from '@/lib/haptics'
import { isProviderSetupErrorMessage } from '@/lib/provider-setup-errors'
import { setSessionYolo } from '@/lib/yolo-session'
import {
  $composerAttachments,
  addComposerAttachment,
  clearComposerAttachments,
  type ComposerAttachment,
  terminalContextBlocksFromDraft
} from '@/store/composer'
import { clearNotifications, notify, notifyError } from '@/store/notifications'
import { requestDesktopOnboarding } from '@/store/onboarding'
import {
  $busy,
  $messages,
  $yoloActive,
  setAwaitingResponse,
  setBusy,
  setMessages,
  setYoloActive
} from '@/store/session'

import type { ClientSessionState, ImageAttachResponse, SlashExecResponse } from '../../types'

function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()

    reader.addEventListener('load', () => {
      if (typeof reader.result === 'string') {
        resolve(reader.result)
      } else {
        reject(new Error('Could not read recorded audio'))
      }
    })
    reader.addEventListener('error', () => reject(reader.error || new Error('Could not read recorded audio')))
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

interface PromptActionsOptions {
  activeSessionId: string | null
  activeSessionIdRef: MutableRefObject<string | null>
  busyRef: MutableRefObject<boolean>
  branchCurrentSession: () => Promise<boolean>
  createBackendSessionForSend: (preview?: string | null) => Promise<string | null>
  handleSkinCommand: (arg: string) => string
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
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

function renderCommandsCatalog(catalog: CommandsCatalogLike): string {
  const desktopCatalog = filterDesktopCommandsCatalog(catalog)

  const sections = desktopCatalog.categories?.length
    ? desktopCatalog.categories
    : [{ name: 'Desktop commands', pairs: desktopCatalog.pairs ?? [] }]

  const body = sections
    .filter(section => section.pairs.length > 0)
    .map(section => {
      const rows = section.pairs.map(([cmd, desc]) => `${cmd.padEnd(18)} ${desc}`)

      return [`${section.name}:`, ...rows].join('\n')
    })
    .join('\n\n')

  const tail = [
    desktopCatalog.skill_count ? `${desktopCatalog.skill_count} skill commands available.` : '',
    desktopCatalog.warning ? `warning: ${desktopCatalog.warning}` : ''
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

export function usePromptActions({
  activeSessionId,
  activeSessionIdRef,
  busyRef,
  branchCurrentSession,
  createBackendSessionForSend,
  handleSkinCommand,
  requestGateway,
  selectedStoredSessionIdRef,
  startFreshSessionDraft,
  sttEnabled,
  updateSessionState
}: PromptActionsOptions) {
  const appendSessionTextMessage = useCallback(
    (sessionId: string, role: ChatMessage['role'], text: string) => {
      const body = text.trim()

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

  const syncImageAttachmentsForSubmit = useCallback(
    async (
      sessionId: string,
      attachments: ComposerAttachment[],
      options: { updateComposerAttachments?: boolean } = {}
    ) => {
      const updateComposerAttachments = options.updateComposerAttachments ?? true
      const images = attachments.filter(attachment => attachment.kind === 'image' && attachment.path)

      for (const attachment of images) {
        if (attachment.attachedSessionId === sessionId) {
          continue
        }

        const result = await requestGateway<ImageAttachResponse>('image.attach', {
          session_id: sessionId,
          path: attachment.path
        })

        if (!result.attached) {
          const label = attachment.label || (attachment.path ? pathLabel(attachment.path) : 'image')
          throw new Error(result.message || `Could not attach ${label}`)
        }

        const attachedPath = result.path || attachment.path

        if (updateComposerAttachments) {
          addComposerAttachment({
            ...attachment,
            id: attachment.id,
            label: attachedPath ? pathLabel(attachedPath) : attachment.label,
            path: attachedPath,
            attachedSessionId: sessionId
          })
        }
      }
    },
    [requestGateway]
  )

  const submitPromptText = useCallback(
    async (rawText: string, options?: SubmitTextOptions) => {
      const visibleText = rawText.trim()
      const usingComposerAttachments = !options?.attachments
      const attachments = options?.attachments ?? $composerAttachments.get()

      const contextRefs = attachments
        .map(a => a.refText)
        .filter(Boolean)
        .join('\n')

      const terminalContextBlocks = terminalContextBlocksFromDraft(rawText).join('\n\n')
      const hasImage = attachments.some(a => a.kind === 'image')
      const attachmentRefs = attachments.map(attachmentDisplayText).filter((r): r is string => Boolean(r))

      const text =
        [contextRefs, terminalContextBlocks, visibleText].filter(Boolean).join('\n\n') ||
        (hasImage ? 'What do you see in this image?' : '')

      if (!text || busyRef.current) {
        return false
      }

      const optimisticId = `user-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`

      const userMessage: ChatMessage = {
        id: optimisticId,
        role: 'user',
        parts: [textPart(visibleText || (attachmentRefs.length ? '' : attachments.map(a => a.label).join(', ')))],
        attachmentRefs
      }

      const releaseBusy = () => {
        busyRef.current = false
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
              : [...state.messages, userMessage],
            busy: true,
            awaitingResponse: true,
            pendingBranchGroup: null,
            sawAssistantPayload: false,
            interrupted: state.interrupted
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

      busyRef.current = true
      setBusy(true)
      setAwaitingResponse(true)
      clearNotifications()

      let sessionId: null | string = activeSessionId

      if (sessionId) {
        seedOptimistic(sessionId)
      } else {
        setMessages(current => [...current, userMessage])
      }

      if (!sessionId) {
        try {
          sessionId = await createBackendSessionForSend(visibleText)
        } catch (err) {
          dropOptimistic(null)
          releaseBusy()
          notifyError(err, 'Session unavailable')

          return false
        }

        if (!sessionId) {
          dropOptimistic(null)
          releaseBusy()
          notify({ kind: 'error', title: 'Session unavailable', message: 'Could not create a new session' })

          return false
        }

        seedOptimistic(sessionId)
      }

      try {
        await syncImageAttachmentsForSubmit(sessionId, attachments, {
          updateComposerAttachments: usingComposerAttachments
        })
        await requestGateway('prompt.submit', { session_id: sessionId, text })

        if (usingComposerAttachments) {
          clearComposerAttachments()
        }

        return true
      } catch (err) {
        const message = inlineErrorMessage(err, 'Prompt failed')

        releaseBusy()
        updateSessionState(sessionId, state => ({
          ...state,
          messages: [
            ...state.messages,
            {
              id: `assistant-error-${Date.now()}`,
              role: 'assistant',
              parts: [],
              error: message || 'Prompt failed',
              branchGroupId: state.pendingBranchGroup ?? undefined
            }
          ],
          busy: false,
          awaitingResponse: false,
          pendingBranchGroup: null,
          sawAssistantPayload: true
        }))

        if (isProviderSetupError(err)) {
          requestDesktopOnboarding('Add a provider credential before sending your first message.')

          return false
        }

        notifyError(err, 'Prompt failed')

        return false
      }
    },
    [
      activeSessionId,
      busyRef,
      createBackendSessionForSend,
      requestGateway,
      selectedStoredSessionIdRef,
      syncImageAttachmentsForSubmit,
      updateSessionState
    ]
  )

  const executeSlashCommand = useCallback(
    async (rawCommand: string, options?: { sessionId?: string; recordInput?: boolean }) => {
      const runSlash = async (commandText: string, sessionHint?: string, recordInput = true): Promise<void> => {
        const command = commandText.trim()
        const { name, arg } = parseSlashCommand(command)
        const normalizedName = name.toLowerCase()

        if (!name) {
          const sessionId = sessionHint || activeSessionIdRef.current || (await createBackendSessionForSend())

          if (sessionId) {
            appendSessionTextMessage(sessionId, 'system', 'empty slash command')
          }

          return
        }

        if (normalizedName === 'new' || normalizedName === 'reset') {
          startFreshSessionDraft()

          return
        }

        if (normalizedName === 'branch' || normalizedName === 'fork') {
          await branchCurrentSession()

          return
        }

        // /yolo maps to the status-bar YOLO control — a per-session approval
        // bypass, same scope as the TUI's Shift+Tab. With no session yet we arm
        // it locally; the session-create path applies it on the first message.
        if (normalizedName === 'yolo') {
          const sid = sessionHint || activeSessionIdRef.current
          const next = !$yoloActive.get()

          if (!sid) {
            setYoloActive(next)
            notify({ kind: 'success', message: next ? 'YOLO armed for this chat' : 'YOLO off' })

            return
          }

          try {
            const active = await setSessionYolo(requestGateway, sid, next)
            appendSessionTextMessage(sid, 'system', `YOLO ${active ? 'on' : 'off'} for this session`)
          } catch {
            notify({ kind: 'error', title: 'YOLO', message: 'Could not toggle YOLO' })
          }

          return
        }

        if (normalizedName === 'skin' && !sessionHint && !activeSessionIdRef.current) {
          notify({ kind: 'success', message: handleSkinCommand(arg) })

          return
        }

        const sessionId = sessionHint || activeSessionIdRef.current || (await createBackendSessionForSend())

        if (!sessionId) {
          notify({
            kind: 'error',
            title: 'Session unavailable',
            message: 'Could not create a new session'
          })

          return
        }

        const renderSlashOutput = (text: string) =>
          appendSessionTextMessage(sessionId, 'system', recordInput ? slashStatusText(command, text) : text)

        if (normalizedName === 'skin') {
          renderSlashOutput(handleSkinCommand(arg))

          return
        }

        if (name === 'help' || name === 'commands') {
          try {
            const catalog = await requestGateway<CommandsCatalogLike>('commands.catalog', { session_id: sessionId })

            renderSlashOutput(renderCommandsCatalog(catalog))
          } catch (err) {
            renderSlashOutput(`error: ${err instanceof Error ? err.message : String(err)}`)
          }

          return
        }

        if (!isDesktopSlashCommand(name)) {
          renderSlashOutput(desktopSlashUnavailableMessage(name) || `/${name} is not available in the desktop app.`)

          return
        }

        try {
          const result = await requestGateway<SlashExecResponse>('slash.exec', {
            session_id: sessionId,
            command: command.replace(/^\/+/, '')
          })

          const body = result?.output || `/${name}: no output`
          renderSlashOutput(result?.warning ? `warning: ${result.warning}\n${body}` : body)

          return
        } catch {
          // Fall back to command.dispatch for skill/send/alias directives.
        }

        try {
          const dispatch = parseCommandDispatch(
            await requestGateway<unknown>('command.dispatch', {
              session_id: sessionId,
              name,
              arg
            })
          )

          if (!dispatch) {
            renderSlashOutput('error: invalid response: command.dispatch')

            return
          }

          if (dispatch.type === 'exec' || dispatch.type === 'plugin') {
            renderSlashOutput(dispatch.output ?? '(no output)')

            return
          }

          if (dispatch.type === 'alias') {
            await runSlash(`/${dispatch.target}${arg ? ` ${arg}` : ''}`, sessionId, false)

            return
          }

          const message = ('message' in dispatch ? dispatch.message : '')?.trim() ?? ''

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
        } catch (err) {
          renderSlashOutput(`error: ${err instanceof Error ? err.message : String(err)}`)
        }
      }

      await runSlash(rawCommand, options?.sessionId, options?.recordInput ?? true)
    },
    [
      activeSessionIdRef,
      appendSessionTextMessage,
      branchCurrentSession,
      busyRef,
      createBackendSessionForSend,
      handleSkinCommand,
      requestGateway,
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
        throw new Error('Speech-to-text is disabled in settings.')
      }

      const dataUrl = await blobToDataUrl(audio)
      const result = await transcribeAudio(dataUrl, audio.type)

      return result.transcript
    },
    [sttEnabled]
  )

  const cancelRun = useCallback(async () => {
    const sessionId = activeSessionId || activeSessionIdRef.current

    busyRef.current = false
    setBusy(false)
    setAwaitingResponse(false)

    const finalizeMessages = (messages: ChatMessage[]) =>
      messages.map(message =>
        message.pending
          ? {
              ...message,
              parts: chatMessageText(message).trim()
                ? appendTextPart(message.parts, INTERRUPTED_MARKER)
                : [...message.parts, textPart(INTERRUPTED_MARKER.trim())],
              pending: false
            }
          : message
      )

    if (!sessionId) {
      setMessages(finalizeMessages($messages.get()))

      return
    }

    updateSessionState(sessionId, state => {
      const streamId = state.streamId

      const messages = streamId
        ? state.messages.map(message =>
            message.id === streamId
              ? {
                  ...message,
                  parts: chatMessageText(message).trim()
                    ? appendTextPart(message.parts, INTERRUPTED_MARKER)
                    : [...message.parts, textPart(INTERRUPTED_MARKER.trim())],
                  pending: false
                }
              : message
          )
        : finalizeMessages(state.messages)

      return {
        ...state,
        messages,
        busy: false,
        awaitingResponse: false,
        streamId: null,
        pendingBranchGroup: null,
        interrupted: true
      }
    })

    try {
      await requestGateway('session.interrupt', { session_id: sessionId })
    } catch (err) {
      notifyError(err, 'Stop failed')
    }
  }, [activeSessionId, activeSessionIdRef, busyRef, requestGateway, updateSessionState])

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
        notifyError(err, 'Regenerate failed')
      }
    },
    [activeSessionId, requestGateway, updateSessionState]
  )

  const editMessage = useCallback(
    async (edited: AppendMessage) => {
      const sessionId = activeSessionId || activeSessionIdRef.current
      const sourceId = edited.sourceId || edited.parentId
      const text = appendText(edited)

      if (!sessionId || !sourceId || !text || edited.role !== 'user' || $busy.get()) {
        return
      }

      const messages = $messages.get()
      const sourceIndex = messages.findIndex(m => m.id === sourceId)
      const source = messages[sourceIndex]

      if (!source || source.role !== 'user' || chatMessageText(source).trim() === text) {
        return
      }

      // Failed turn: optimistic user msg never reached the gateway, so truncating
      // by ordinal would 422. Submit as a plain resend instead.
      const nextMessage = messages[sourceIndex + 1]
      const isFailedTurn = nextMessage?.role === 'assistant' && Boolean(nextMessage.error)
      const editedMessage: ChatMessage = { ...source, parts: [textPart(text)] }

      clearNotifications()
      busyRef.current = true
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

      const submit = (truncateOrdinal?: number) =>
        requestGateway('prompt.submit', {
          session_id: sessionId,
          text,
          ...(truncateOrdinal !== undefined && { truncate_before_user_ordinal: truncateOrdinal })
        })

      const isStaleTargetError = (err: unknown) =>
        /no longer in session history|not in session history/i.test(err instanceof Error ? err.message : String(err))

      try {
        await submit(isFailedTurn ? undefined : visibleUserOrdinal(messages, sourceIndex))
      } catch (err) {
        let surfaced = err

        if (!isFailedTurn && isStaleTargetError(err)) {
          try {
            await submit()

            return
          } catch (retryErr) {
            surfaced = retryErr
          }
        }

        busyRef.current = false
        setBusy(false)
        setAwaitingResponse(false)
        updateSessionState(sessionId, state => ({ ...state, busy: false, awaitingResponse: false }))
        notifyError(surfaced, 'Edit failed')
      }
    },
    [activeSessionId, activeSessionIdRef, busyRef, requestGateway, updateSessionState]
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

  return { cancelRun, editMessage, handleThreadMessagesChange, reloadFromMessage, submitText, transcribeVoiceAudio }
}

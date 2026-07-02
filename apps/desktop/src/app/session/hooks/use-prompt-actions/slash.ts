import { type MutableRefObject, useCallback } from 'react'

import { getProfiles } from '@/hermes'
import type { Translations } from '@/i18n'
import { type ChatMessage } from '@/lib/chat-messages'
import { parseCommandDispatch, parseSlashCommand, sessionTitle } from '@/lib/chat-runtime'
import {
  type CommandsCatalogLike,
  type DesktopActionId,
  type DesktopPickerId,
  desktopSlashUnavailableMessage,
  isDesktopSlashCommand,
  resolveDesktopCommand
} from '@/lib/desktop-slash-commands'
import { setSessionYolo } from '@/lib/yolo-session'
import { openCommandPalettePage } from '@/store/command-palette'
import { type ComposerAttachment, setComposerDraft } from '@/store/composer'
import { notify, notifyError } from '@/store/notifications'
import { setPetScale } from '@/store/pet-gallery'
import { $petGenInput, openPetGenerate } from '@/store/pet-generate'
import { $activeGatewayProfile, $newChatProfile, ensureGatewayProfile, normalizeProfileKey } from '@/store/profile'
import {
  $connection,
  $sessions,
  $yoloActive,
  setModelPickerOpen,
  setSessionPickerOpen,
  setSessions,
  setYoloActive
} from '@/store/session'

import type { BrowserManageResponse, SessionTitleResponse, SlashExecResponse } from '../../../types'

import { type GatewayRequest, isSessionIdCandidate, renderCommandsCatalog, slashStatusText } from './utils'

/** Everything a slash handler needs about the invocation it's serving. */
interface SlashActionCtx {
  arg: string
  command: string
  name: string
  recordInput: boolean
  sessionHint?: string
}

interface SlashCommandDeps {
  activeSessionIdRef: MutableRefObject<string | null>
  appendSessionTextMessage: (sessionId: string, role: ChatMessage['role'], text: string) => void
  branchCurrentSession: () => Promise<boolean>
  busyRef: MutableRefObject<boolean>
  copy: Translations['desktop']
  createBackendSessionForSend: (preview?: string | null) => Promise<string | null>
  handleSkinCommand: (arg: string) => string
  handoffSession: (
    platform: string,
    options?: { onProgress?: (state: string) => void; sessionId?: string }
  ) => Promise<{ ok: boolean; error?: string }>
  refreshSessions: () => Promise<void>
  requestGateway: GatewayRequest
  resumeStoredSession: (storedSessionId: string) => Promise<void> | void
  startFreshSessionDraft: () => void
  submitPromptText: (
    rawText: string,
    options?: { attachments?: ComposerAttachment[]; fromQueue?: boolean }
  ) => Promise<boolean>
}

/** The /slash command dispatcher, extracted from usePromptActions. */
export function useSlashCommand(deps: SlashCommandDeps) {
  const {
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
  } = deps

  return useCallback(
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
}

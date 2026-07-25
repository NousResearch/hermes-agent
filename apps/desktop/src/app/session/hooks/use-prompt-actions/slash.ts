import { type MutableRefObject, useCallback, useRef } from 'react'

import { getProfiles } from '@/hermes'
import type { Translations } from '@/i18n'
import { type ChatMessage, toChatMessages } from '@/lib/chat-messages'
import { parseCommandDispatch, parseSlashCommand, sessionTitle } from '@/lib/chat-runtime'
import {
  type CommandsCatalogLike,
  type DesktopActionId,
  type DesktopCommandSurface,
  type DesktopPickerId,
  desktopSlashUnavailableMessage,
  isDesktopSlashCommand,
  resolveDesktopCommand
} from '@/lib/desktop-slash-commands'
import { isMissingRpcMethod } from '@/lib/gateway-rpc'
import { setSessionYolo } from '@/lib/yolo-session'
import { openCommandPalettePage } from '@/store/command-palette'
import { setComposerDraft } from '@/store/composer'
import { dismissNotification, notify, notifyError } from '@/store/notifications'
import { setPetScale } from '@/store/pet-gallery'
import { $petGenInput, openPetGenerate } from '@/store/pet-generate'
import { $activeGatewayProfile, $newChatProfile, ensureGatewayProfile, normalizeProfileKey } from '@/store/profile'
import {
  $connection,
  $sessions,
  $yoloActive,
  setCurrentUsage,
  setModelPickerOpen,
  setSessionPickerOpen,
  setSessions,
  setYoloActive
} from '@/store/session'

import type {
  BrowserManageResponse,
  ClientSessionState,
  SessionCompressResponse,
  SessionTitleResponse,
  SlashExecResponse
} from '../../../types'

import {
  type GatewayRequest,
  isSessionIdCandidate,
  renderCommandsCatalog,
  renderRpcResult,
  slashStatusText,
  type SubmitTextOptions
} from './utils'

// Manual compression is LLM-bound and routinely outlives the desktop's 30s
// default WS request timeout on large sessions — give it the TUI client's
// 120s RPC budget (HERMES_TUI_RPC_TIMEOUT_MS default) instead.
const SESSION_COMPRESS_TIMEOUT_MS = 120_000

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
  appendSessionTextMessage: (
    sessionId: string,
    role: ChatMessage['role'],
    text: string,
    storedSessionId?: string | null
  ) => void
  branchCurrentSession: () => Promise<boolean>
  busyRef: MutableRefObject<boolean>
  copy: Translations['desktop']
  createBackendSessionForSend: (preview?: string | null) => Promise<string | null>
  handleSkinCommand: (arg: string) => string
  handoffSession: (
    platform: string,
    options?: { onProgress?: (state: string) => void; sessionId?: string }
  ) => Promise<{ ok: boolean; error?: string }>
  openMemoryGraph: () => void
  refreshSessions: () => Promise<void>
  requestGateway: GatewayRequest
  resumeStoredSession: (storedSessionId: string) => Promise<void> | void
  selectedStoredSessionIdRef: MutableRefObject<string | null>
  startFreshSessionDraft: () => void
  submitPromptText: (rawText: string, options?: SubmitTextOptions) => Promise<boolean>
  updateSessionState: (
    sessionId: string,
    updater: (state: ClientSessionState) => ClientSessionState,
    storedSessionId?: string | null
  ) => ClientSessionState
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
    openMemoryGraph,
    refreshSessions,
    requestGateway,
    resumeStoredSession,
    selectedStoredSessionIdRef,
    startFreshSessionDraft,
    submitPromptText,
    updateSessionState
  } = deps

  const compressInFlightRef = useRef(new Set<string>())

  return useCallback(
    async (rawCommand: string, options?: { sessionId?: string; recordInput?: boolean }) => {
      const ensureSessionId = async (sessionHint?: string) =>
        sessionHint || activeSessionIdRef.current || (await createBackendSessionForSend())

      // Resolve the target session plus a writer for inline slash output, or
      // notify + return null when none can be created. Folds the ensure / bail /
      // build-renderSlashOutput boilerplate every exec-style handler repeats.
      const withSlashOutput = async (
        ctx: SlashActionCtx
      ): Promise<{ render: (text: string) => void; sessionId: string; storedSessionId: string | null } | null> => {
        const sessionId = await ensureSessionId(ctx.sessionHint)

        if (!sessionId) {
          notify({ kind: 'error', title: copy.sessionUnavailable, message: copy.createSessionFailed })

          return null
        }

        // A long-running command can finish after a session switch. Keep its
        // output bound to the stored session selected at invocation time.
        const storedSessionId = selectedStoredSessionIdRef.current

        const render = (text: string) =>
          appendSessionTextMessage(
            sessionId,
            'system',
            ctx.recordInput ? slashStatusText(ctx.command, text) : text,
            storedSessionId
          )

        return { render, sessionId, storedSessionId }
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

        let slashExecError: unknown = null

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
        } catch (error) {
          // Fall back to command.dispatch for skill/send/alias directives, but
          // keep the worker error: a slash.exec worker timeout/crash is the real
          // failure, not the "not a quick/plugin/skill command" routing noise.
          slashExecError = error
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
          // "not a quick/plugin/skill command" just means the fallback had
          // nothing to add — the slash.exec failure (worker timeout, crash) is
          // the real error, so don't bury it under the routing noise.
          const dispatchMessage = err instanceof Error ? err.message : String(err)

          if (slashExecError && /not a quick\/plugin\/skill command/i.test(dispatchMessage)) {
            const original = slashExecError instanceof Error ? slashExecError.message : String(slashExecError)
            renderSlashOutput(`error: /${name} failed: ${original}`)

            return
          }

          renderSlashOutput(`error: ${dispatchMessage}`)
        }
      }

      // `rpc` commands have a dedicated gateway handler — skip slash.exec /
      // command.dispatch entirely. The response is structured (a JSON RPC
      // reply, not an inline text stream) so we render it via the generic
      // `renderRpcResult` shaper that knows the field conventions shared by
      // session.save / session.status / session.usage / session.steer /
      // process.stop / agents.list.
      async function runRpc(
        surface: Extract<DesktopCommandSurface, { kind: 'rpc' }>,
        ctx: SlashActionCtx
      ): Promise<void> {
        const resolved = await withSlashOutput(ctx)

        if (!resolved) {
          return
        }

        const { render: renderSlashOutput, sessionId } = resolved

        try {
          const params = surface.buildParams({
            arg: ctx.arg,
            command: ctx.command,
            name: ctx.name,
            sessionId
          })

          // Forward the surface's declared timeout when present; the default
          // requestGateway layer keeps (30s) is too tight for RPCs that do
          // real work.
          const result = await requestGateway<unknown>(surface.rpc, params, surface.timeoutMs)
          const body = renderRpcResult(result, ctx.name)

          renderSlashOutput(body || `/${ctx.name}: no output`)
        } catch (err) {
          // TODO: remove this compatibility fallback once every supported
          // managed runtime exposes the dedicated RPC surface. Desktop and its
          // gateway update independently; older gateways still support the
          // slash-worker route.
          if (isMissingRpcMethod(err)) {
            await runExec(ctx)

            return
          }

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
        // /compress (alias /compact) runs the gateway's dedicated
        // session.compress RPC — the TUI's path
        // (ui-tui/src/app/slash/commands/session.ts). It must NOT go through
        // runExec: compressing a large session outlives the slash worker's pipe
        // timeout (45s) and the desktop's 30s WS default, and the resulting
        // slash.exec error cascaded into command.dispatch's misleading "not a
        // quick/plugin/skill command: compress" (#44456).
        //
        // The RPC returns the post-compress `messages` array (same shape
        // session.resume returns), so we replace the transcript from it —
        // otherwise the summarized bubbles stay on screen forever (#44462
        // review). `updateSessionState` only publishes for the active runtime,
        // so a late result after a session switch refreshes its own cache
        // without clobbering the foreground transcript (#53755 review).
        compress: async ctx => {
          const resolved = await withSlashOutput(ctx)

          if (!resolved) {
            return
          }

          const { render: renderSlashOutput, sessionId, storedSessionId } = resolved
          const focusTopic = ctx.arg.trim()
          const noticeId = `session-compress:${sessionId}`

          // Coalesce concurrent compress requests for the same session so a
          // double-enter doesn't fire two LLM summarise calls.
          if (compressInFlightRef.current.has(sessionId)) {
            return
          }

          compressInFlightRef.current.add(sessionId)
          notify({
            durationMs: 0,
            id: noticeId,
            kind: 'info',
            message: focusTopic ? `compressing context for: ${focusTopic}` : 'compressing context...'
          })

          try {
            const result = await requestGateway<SessionCompressResponse>(
              'session.compress',
              {
                session_id: sessionId,
                ...(focusTopic ? { focus_topic: focusTopic } : {})
              },
              SESSION_COMPRESS_TIMEOUT_MS
            )

            // Replace the transcript with the post-compress history so the
            // summarized bubbles actually disappear. `messages` is the same
            // shape session.resume returns (_history_to_messages), so
            // toChatMessages handles it directly. updateSessionState only
            // publishes for the active runtime, guarding against a late result
            // clobbering the foreground after a session switch.
            if (Array.isArray(result?.messages)) {
              updateSessionState(
                sessionId,
                state => ({ ...state, messages: toChatMessages(result.messages!) }),
                storedSessionId
              )
            }

            const usage = { ...result?.usage, ...result?.info?.usage }

            if (Object.keys(usage).length && activeSessionIdRef.current === sessionId) {
              setCurrentUsage(current => ({ ...current, ...usage }))
            }

            if (result?.info?.title !== undefined) {
              setSessions(prev =>
                prev.map(session =>
                  session.id === sessionId ? { ...session, title: result.info!.title || null } : session
                )
              )
            }

            if (result?.summary?.headline) {
              const lines = [result.summary.headline, result.summary.token_line, result.summary.note].filter(
                (line): line is string => Boolean(line)
              )

              const aborted = result.status === 'aborted' || result.summary.aborted === true

              if (!aborted) {
                // Keep a durable record of a successful manual compression in
                // the chat after replacing it with the authoritative backend
                // transcript. Errors remain transient: appending an error as a
                // system message would look like a successful state change.
                renderSlashOutput(lines.join('\n'))
              }

              notify({
                durationMs: 5_000,
                id: noticeId,
                kind: aborted ? 'error' : 'success',
                message: lines.join('\n')
              })

              return
            }

            const hostOutput = result?.host_ack?.output?.trim()

            if (hostOutput) {
              renderSlashOutput(hostOutput)
              notify({ durationMs: 5_000, id: noticeId, kind: 'success', message: hostOutput })

              return
            }

            const removed = result?.removed ?? 0
            const message = removed > 0 ? `compressed ${removed} messages` : 'nothing to compress'
            renderSlashOutput(message)
            notify({
              durationMs: 5_000,
              id: noticeId,
              kind: 'success',
              message
            })
          } catch (err) {
            dismissNotification(noticeId)

            // Desktop and gateway runtimes update independently. Preserve the
            // historical slash-worker path for an older gateway that has not
            // shipped session.compress yet; it cannot offer the longer RPC
            // timeout, but it must not turn a once-working command into an
            // immediate "method not found" error.
            if (isMissingRpcMethod(err)) {
              await runExec(ctx)

              return
            }

            renderSlashOutput(`error: ${err instanceof Error ? err.message : String(err)}`)
          } finally {
            compressInFlightRef.current.delete(sessionId)
          }
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
        // /journey (aliases /learning, /memory-graph) opens the memory graph
        // overlay — the desktop's visual counterpart of the TUI journey
        // timeline — instead of printing a text rendering into the transcript.
        // Args are ignored, matching the TUI overlay behavior.
        journey: async () => {
          openMemoryGraph()
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
          // The composer draft was already cleared on submit, and slash input
          // never lands in the Up-arrow history ring (it derives from sent user
          // messages) — so without this restore, any payload after a degenerate
          // slash (`/ text`, `/` + newline) is lost forever. Hand it back.
          if (command.replace(/^\/+/, '').trim()) {
            setComposerDraft(command)
          }

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

          case 'rpc':
            return runRpc(surface, ctx)

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
      openMemoryGraph,
      refreshSessions,
      requestGateway,
      resumeStoredSession,
      selectedStoredSessionIdRef,
      startFreshSessionDraft,
      submitPromptText,
      updateSessionState
    ]
  )
}

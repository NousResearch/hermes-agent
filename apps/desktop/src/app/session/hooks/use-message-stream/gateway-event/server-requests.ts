import { readActivePreview } from '@/app/chat/right-rail/preview-reader'
import { readActiveTerminal } from '@/app/right-sidebar/terminal/buffer'
import { pendingClarifyToolPayload } from '@/app/session/hooks/use-session-actions/restore-pending-clarify'
import { translateNow } from '@/i18n'
import { restorePendingClarifyToolCall } from '@/lib/chat-messages'
import type { PreviewActAction } from '@/lib/preview-act/act-in-page'
import type { TourAction, TourStep } from '@/lib/tour'
import { normalizeChoices, normalizeQuestions, setClarifyRequest, warnDroppedChoices } from '@/store/clarify'
import type { ScopedServerRequest } from '@/store/gateway'
import { $hudMode } from '@/store/hud'
import { dispatchNativeNotification } from '@/store/native-notifications'
import {
  receiveApprovalRequest,
  setSecretRequest,
  setSudoRequest,
  setVaultCodeRequest,
  setVaultSaveLoginRequest,
  setVaultUnlockRequest
} from '@/store/prompts'
import { rememberServerRequest } from '@/store/server-requests'
import { $selectedStoredSessionId, $sessions, sessionMatchesStoredId } from '@/store/session'
import { $sessionTiles, storedSessionIdForRuntimeId } from '@/store/session-states'
import { requestScrollToBottom } from '@/store/thread-scroll'
import { $toursEnabled } from '@/store/tours'

import type { GatewayEventDeps } from './types'

/** The preview engine, loaded on demand so ~25KB of page-injectable source stays
 *  off the boot path (dev: a fresh copy per action so edits reach the guest — see
 *  the previous home of this loader in desktop-bridge.ts for the full story). */
const loadPreviewEngine = () => {
  const stable = () => import('@/app/chat/right-rail/preview-act')

  if (!import.meta.hot) {
    return stable().then(mod => mod.actOnActivePreview)
  }

  return import(/* @vite-ignore */ '/src/app/chat/right-rail/preview-act.ts?hot=' + Date.now())
    .catch(stable)
    .then(mod => mod.actOnActivePreview as Awaited<ReturnType<typeof stable>>['actOnActivePreview'])
}

const str = (v: unknown): string => (typeof v === 'string' ? v : '')
const num = (v: unknown): number | undefined => (typeof v === 'number' ? v : undefined)

/** Answer a string-valued request with a JSON-encoded result ('' = nothing / unavailable). */
const answerValue = (request: ScopedServerRequest, result: unknown) =>
  request.respond({ value: result ? JSON.stringify(result) : '' })

export interface ServerRequestContext {
  deps: Pick<GatewayEventDeps, 'activeSessionIdRef' | 'sessionInterrupted' | 'updateSessionState' | 'upsertToolCall'>
  request: ScopedServerRequest
  /** The session the request names ('' when unscoped). */
  sessionId: string
  /** The named session is the one on screen. */
  isActiveSession: boolean
}

type Handler = (ctx: ServerRequestContext) => void

type PreviewSessionRoute = 'ignore' | 'retry' | 'run'

/**
 * Bridges answered from THIS window's panes (preview tab, xterm buffer, the
 * native window below, the tour overlay). If one reaches a window that does
 * not host the session, its empty answer could beat the owner's answer and
 * make the tool report "no preview tab / no terminal" while that pane is open
 * (#113348).
 */
const WINDOW_OWNED_REQUESTS = new Set(['preview.act', 'preview.read', 'terminal.read', 'window.read', 'tour'])

/** This window hosts the session: it is the primary view or an open session tile. */
export function windowHostsSession(sessionId: string, activeSessionId: null | string): boolean {
  return sessionId === activeSessionId || $sessionTiles.get().some(tile => tile.runtimeId === sessionId)
}

/**
 * The HUD owns native-window geometry for the conversation it is showing.
 * Its gateway transport receives the scoped request, but the HUD deliberately
 * keeps no `$activeSessionId` view binding of its own; translate the request's
 * runtime id and match its durable selection through the compression lineage.
 */
function hudHostsWindowRead(sessionId: string): boolean {
  if (!$hudMode.get()) {
    return false
  }

  const selectedSessionId = $selectedStoredSessionId.get()

  if (!selectedSessionId) {
    return false
  }

  const storedSessionId = storedSessionIdForRuntimeId(sessionId) ?? sessionId

  return (
    selectedSessionId === storedSessionId ||
    $sessions
      .get()
      .some(
        session =>
          sessionMatchesStoredId(session, selectedSessionId) && sessionMatchesStoredId(session, storedSessionId)
      )
  )
}

/**
 * Panes are local to one desktop window. A scoped request may only be answered
 * by the window hosting its session (primary view or a tile). During reconnect,
 * however, an open request can replay one event-loop turn before the resumed
 * session becomes active; retry that one narrow race and otherwise leave the
 * request for its owner.
 */
export function previewSessionRoute({
  activeSessionId,
  replayed,
  sessionId
}: {
  activeSessionId: null | string
  replayed: boolean | undefined
  sessionId: string
}): PreviewSessionRoute {
  if (!sessionId || windowHostsSession(sessionId, activeSessionId)) {
    return 'run'
  }

  return replayed && !activeSessionId ? 'retry' : 'ignore'
}

function windowOwnedSessionRoute(
  method: string,
  route: Parameters<typeof previewSessionRoute>[0]
): PreviewSessionRoute {
  if (method === 'window.read' && hudHostsWindowRead(route.sessionId)) {
    return 'run'
  }

  return previewSessionRoute(route)
}

const markNeedsInput = (ctx: ServerRequestContext) => {
  if (ctx.sessionId) {
    ctx.deps.updateSessionState(ctx.sessionId, state => ({ ...state, needsInput: true }))
  }
}

const notifyInput = (ctx: ServerRequestContext, body: string) => {
  if (!ctx.request.replayed) {
    dispatchNativeNotification({
      body,
      kind: 'input',
      sessionId: ctx.sessionId || null,
      title: translateNow('notifications.native.inputTitle')
    })
  }
}

// ── Blocking-input family (clarify / approval / sudo / secret / vault / MCP setup) ──
// Every one is parked per-session (like clarify) so a BACKGROUND session's turn can
// raise it and wait — the sidebar flags "needs input" and the card surfaces once the
// user focuses that chat. The Python side blocks on the response frame; without a
// handler the channel answers -32601 and the tool fails fast instead of stalling.

const clarify: Handler = ctx => {
  const { deps, request, sessionId } = ctx
  const p = request.params

  if (sessionId && deps.sessionInterrupted(sessionId)) {
    request.respond({ answer: '' })

    return
  }

  const question = str(p.question)
  const rawChoices = p.choices
  const choices = normalizeChoices(rawChoices)
  const multiSelect = p.multi_select === true
  // Batch (multi-question) clarify: `questions` replaces question/choices on the
  // wire. `answers` rides along only on a reconnect replay (locks the server
  // already accepted).
  const questions = normalizeQuestions(p.questions)

  const lockedAnswers =
    typeof p.answers === 'object' && p.answers !== null
      ? Object.fromEntries(
          Object.entries(p.answers as Record<string, unknown>).filter(
            (entry): entry is [string, string] => typeof entry[1] === 'string'
          )
        )
      : undefined

  if (questions.length === 0 && !question) {
    request.respond({ answer: '' })

    return
  }

  if (questions.length === 0 && rawChoices != null && choices.length === 0) {
    warnDroppedChoices('gateway', question, rawChoices)
  }

  const clarifyRequest =
    questions.length > 0
      ? {
          choices: null,
          lockedAnswers,
          multiSelect: false,
          question: '',
          questions,
          receivedAt: Date.now() / 1000,
          requestId: request.id,
          sessionId: sessionId || null
        }
      : {
          choices: choices.length > 0 ? choices : null,
          multiSelect,
          question,
          receivedAt: Date.now() / 1000,
          requestId: request.id,
          sessionId: sessionId || null
        }

  rememberServerRequest(request)
  setClarifyRequest(clarifyRequest)

  if (sessionId) {
    // A resumed/hydrated transcript may already contain this provider's clarify
    // call while carrying no live streamId. Re-arm that exact row instead of
    // letting the generic stream mutator append a second card.
    const occurredAt = Date.now() / 1000

    deps.updateSessionState(sessionId, state => {
      const projection = restorePendingClarifyToolCall(
        state.messages,
        pendingClarifyToolPayload(clarifyRequest),
        occurredAt
      )

      return {
        ...state,
        messages: projection.messages,
        streamId: projection.streamId,
        sawAssistantPayload: true,
        awaitingResponse: false,
        needsInput: true
      }
    })

    if (sessionId === deps.activeSessionIdRef.current) {
      requestScrollToBottom(sessionId)
    }
  }

  notifyInput(ctx, questions.length > 0 ? questions.map(q => q.question).join(' · ') : question)
}

const approval: Handler = ctx => {
  const { request, sessionId } = ctx
  const p = request.params
  const command = str(p.command)
  const description = str(p.description) || 'dangerous command'

  rememberServerRequest(request)
  void receiveApprovalRequest(null, {
    // false only when a tirith warning forbids it; backend omits the field otherwise.
    allowPermanent: p.allow_permanent !== false,
    choices: Array.isArray(p.choices)
      ? p.choices.filter((choice): choice is string => typeof choice === 'string')
      : undefined,
    command,
    description,
    // The approval queue's own id — `approval.pending` / `approval.received` / `approval.respond` key on it.
    requestId: str(p.request_id) || undefined,
    serverRequestId: request.id,
    sessionId: sessionId || null,
    smartDenied: p.smart_denied === true
  }).catch(() => undefined)
  markNeedsInput(ctx)

  if (!request.replayed) {
    dispatchNativeNotification({
      actions: [
        {
          id: str(p.request_id) ? `approve:${str(p.request_id)}` : 'approve',
          text: translateNow('notifications.native.approveAction')
        },
        {
          id: str(p.request_id) ? `reject:${str(p.request_id)}` : 'reject',
          text: translateNow('notifications.native.rejectAction')
        }
      ],
      body: command || description,
      kind: 'approval',
      sessionId: sessionId || null,
      title: translateNow('notifications.native.approvalTitle')
    })
  }
}

const sudo: Handler = ctx => {
  rememberServerRequest(ctx.request)
  setSudoRequest({
    command: str(ctx.request.params.command),
    requestId: ctx.request.id,
    sessionId: ctx.sessionId || null
  })
  markNeedsInput(ctx)
  notifyInput(ctx, translateNow('notifications.native.inputBody'))
}

/** Bot Screen package install (`tui_gateway/methods_display.py`): the same masked card as `sudo`,
 *  but app-level. The gateway sends it sessionless — it belongs to the connection that clicked
 *  Install, not to a chat — so it is stored under the null session and survives a chat switch. */
const displayInstallSudo: Handler = ctx => {
  rememberServerRequest(ctx.request)
  setSudoRequest({
    description: translateNow('prompts.sudoInstallDesc'),
    requestId: ctx.request.id,
    sessionId: null
  })
  notifyInput(ctx, translateNow('prompts.sudoInstallDesc'))
}

const secret: Handler = ctx => {
  const p = ctx.request.params
  const envVar = str(p.env_var)
  const promptText = str(p.prompt)

  rememberServerRequest(ctx.request)
  setSecretRequest({ envVar, prompt: promptText, requestId: ctx.request.id, sessionId: ctx.sessionId || null })
  markNeedsInput(ctx)
  notifyInput(ctx, promptText || envVar || translateNow('notifications.native.inputBody'))
}

const vaultCode: Handler = ctx => {
  const p = ctx.request.params
  const site = str(p.site)

  rememberServerRequest(ctx.request)
  setVaultCodeRequest({ hint: str(p.hint), requestId: ctx.request.id, sessionId: ctx.sessionId || null, site })
  markNeedsInput(ctx)
  notifyInput(ctx, translateNow('prompts.vaultCodeTitle', site))
}

const vaultSaveLogin: Handler = ctx => {
  const p = ctx.request.params
  const origin = str(p.origin)
  const site = str(p.site) || origin

  rememberServerRequest(ctx.request)
  setVaultSaveLoginRequest({ origin, requestId: ctx.request.id, sessionId: ctx.sessionId || null, site })
  markNeedsInput(ctx)
  notifyInput(ctx, translateNow('prompts.vaultSaveTitle', site))
}

const vaultUnlockPrompt: Handler = ctx => {
  const p = ctx.request.params
  const backend = str(p.backend)
  const displayName = str(p.display_name) || backend

  rememberServerRequest(ctx.request)
  setVaultUnlockRequest({ backend, displayName, requestId: ctx.request.id, sessionId: ctx.sessionId || null })
  markNeedsInput(ctx)
  notifyInput(ctx, translateNow('prompts.vaultUnlockTitle', displayName))
}

// ── Desktop-surface bridges (answered immediately, no card) ─────────────────

const terminalRead: Handler = ({ request }) => {
  // read_terminal tool: serialize the renderer's xterm buffer. Empty = no live pane.
  answerValue(request, readActiveTerminal({ count: num(request.params.count), start: num(request.params.start) }))
}

const previewRead: Handler = ({ request }) => {
  // read_preview tool: the active preview tab's page text is async. Empty = nothing open.
  void readActivePreview({ count: num(request.params.count), start: num(request.params.start) }).then(result =>
    answerValue(request, result)
  )
}

const previewAct: Handler = ({ isActiveSession, request }) => {
  // drive_preview tool: click/type/scroll/press inside the guest page. Active
  // session only: a background turn (including one in a tile this window hosts)
  // must never reach into the page the user is working in (desktop AGENTS.md:
  // offer, don't hijack). Window ownership is settled by WINDOW_OWNED_REQUESTS
  // before this runs, so a refusal here reaches the tool instead of stalling it.
  const p = request.params

  if (!isActiveSession) {
    answerValue(request, {
      error: 'The in-app browser only takes actions in the session the user is looking at.',
      success: false
    })

    return
  }

  void loadPreviewEngine()
    .then(run =>
      run({
        amount: p.amount as never,
        key: p.key as never,
        kind: (str(p.action) || '') as never,
        max: p.max as never,
        ref: p.ref as never,
        selector: p.selector as never,
        submit: p.submit as never,
        text: p.text as never,
        to: p.to as PreviewActAction['to']
      })
    )
    .then(
      result => answerValue(request, result),
      error => answerValue(request, { error: error instanceof Error ? error.message : String(error), success: false })
    )
}

const windowRead: Handler = ({ request }) => {
  // read_window_below tool: main owns native window enumeration. Empty =
  // unavailable (older shell without the handler, Wayland, …) — without an
  // answer the tool would stall its full 30s deadline.
  const read = window.hermesDesktop?.readWindowBelow

  void Promise.resolve(read ? read() : null).then(
    result => answerValue(request, result),
    () => answerValue(request, null)
  )
}

const tour: Handler = ({ isActiveSession, request }) => {
  // tour tool: one guided-tour action via driver.js, app DOM or preview guest
  // page. Active session only, same window-ownership rule as preview.act
  // (WINDOW_OWNED_REQUESTS).
  const p = request.params

  if (!$toursEnabled.get()) {
    // Refused in words, not silently dropped: a no-op would leave the agent
    // narrating a spotlight the user can't see.
    answerValue(request, { error: 'The user has turned guided tours off.', success: false })

    return
  }

  if (!isActiveSession) {
    answerValue(request, { error: 'Tours only run in the session the user is looking at.', success: false })

    return
  }

  void import('@/lib/tour')
    .then(({ runTour }) =>
      runTour(
        {
          kind: (str(p.action) || 'stop') as TourAction['kind'],
          selector: p.selector as never,
          side: p.side as TourStep['side'],
          startAt: p.step_index as never,
          steps: p.steps as TourStep[] | undefined,
          text: p.text as never,
          title: p.title as never
        },
        p.surface === 'preview' ? 'preview' : 'app'
      )
    )
    .then(
      result => answerValue(request, result),
      error => answerValue(request, { error: error instanceof Error ? error.message : String(error), success: false })
    )
}

/** Method → handler. Every `ServerRequestMap` key the desktop answers. */
export const SERVER_REQUEST_HANDLERS: Record<string, Handler> = {
  approval,
  clarify,
  'display.install.sudo': displayInstallSudo,
  'preview.act': previewAct,
  'preview.read': previewRead,
  secret,
  sudo,
  'terminal.read': terminalRead,
  tour,
  'vault.code': vaultCode,
  'vault.save_login': vaultSaveLogin,
  'vault.unlock_prompt': vaultUnlockPrompt,
  'window.read': windowRead
}

/** Dispatch one server→client request; false when the desktop has no handler for its method. */
export function handleServerRequest(
  request: ScopedServerRequest,
  deps: ServerRequestContext['deps'],
  activeSessionId: null | string
): boolean {
  const handler = SERVER_REQUEST_HANDLERS[request.method]

  if (!handler) {
    return false
  }

  const sessionId = str(request.params.session_id)

  if (WINDOW_OWNED_REQUESTS.has(request.method)) {
    const route = windowOwnedSessionRoute(request.method, {
      activeSessionId,
      replayed: request.replayed,
      sessionId
    })

    if (route === 'ignore') {
      return true
    }

    if (route === 'retry') {
      // Re-read the ref instead of capturing activeSessionId: session resume
      // publishes its binding synchronously between this replay and the next
      // turn. A second miss deliberately stays silent for another window.
      setTimeout(() => {
        if (
          windowOwnedSessionRoute(request.method, {
            activeSessionId: deps.activeSessionIdRef.current,
            replayed: false,
            sessionId
          }) === 'run'
        ) {
          handler({ deps, request, sessionId, isActiveSession: true })
        }
      }, 0)

      return true
    }
  }

  handler({ deps, request, sessionId, isActiveSession: Boolean(sessionId) && sessionId === activeSessionId })

  return true
}

import type { ClarifyParams, PreviewActParams, ServerRequestMap, TourParams, TourStep as WireTourStep } from '@hermes/shared'

import { type PreviewReadResult, readActivePreview } from '@/app/chat/right-rail/preview-reader'
import { readActiveTerminal, type TerminalReadResult } from '@/app/right-sidebar/terminal/buffer'
import { pendingClarifyToolPayload } from '@/app/session/hooks/use-session-actions/restore-pending-clarify'
import { translateNow } from '@/i18n'
import { restorePendingClarifyToolCall } from '@/lib/chat-messages'
import type { PreviewActAction, PreviewActResult } from '@/lib/preview-act/act-in-page'
import type { TourAction, TourResult, TourStep } from '@/lib/tour'
import { type ClarifyRequest, displayChoices, setClarifyRequest } from '@/store/clarify'
import type { ScopedServerRequest } from '@/store/gateway'
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
    // SAFETY: the hot URL serves the same module as `stable`, only cache-busted.
    .then(mod => mod.actOnActivePreview as Awaited<ReturnType<typeof stable>>['actOnActivePreview'])
}

export interface ServerRequestContext<M extends keyof ServerRequestMap = keyof ServerRequestMap> {
  deps: Pick<GatewayEventDeps, 'activeSessionIdRef' | 'sessionInterrupted' | 'updateSessionState' | 'upsertToolCall'>
  request: ScopedServerRequest<M>
  /** The session the request names. */
  sessionId: string
  /** The named session is the one on screen. */
  isActiveSession: boolean
}

type Handler<M extends keyof ServerRequestMap> = (ctx: ServerRequestContext<M>) => void

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
// user focuses that chat. The Python side blocks on the response frame.

/** The card to park for a clarify, or null when there is nothing to ask. */
const clarifyRequestFrom = (params: ClarifyParams, requestId: string, sessionId: string): ClarifyRequest | null => {
  const parked = { receivedAt: Date.now() / 1000, requestId, sessionId: sessionId || null }

  if (params.kind === 'batch') {
    // `answers` rides along only on a reconnect replay (locks the server already accepted).
    return params.questions.length > 0
      ? {
          ...parked,
          kind: 'batch',
          lockedAnswers: params.answers,
          questions: params.questions.map(question => ({ ...question, choices: displayChoices(question.choices) }))
        }
      : null
  }

  return params.question
    ? {
        ...parked,
        choices: displayChoices(params.choices),
        kind: 'single',
        multi_select: params.multi_select,
        question: params.question
      }
    : null
}

const clarify: Handler<'clarify'> = ctx => {
  const { deps, request, sessionId } = ctx

  if (sessionId && deps.sessionInterrupted(sessionId)) {
    request.respond({ answer: '' })

    return
  }

  const clarifyRequest = clarifyRequestFrom(request.params, request.id, sessionId)

  if (!clarifyRequest) {
    request.respond({ answer: '' })

    return
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

  notifyInput(
    ctx,
    clarifyRequest.kind === 'batch'
      ? clarifyRequest.questions.map(question => question.question).join(' · ')
      : clarifyRequest.question
  )
}

const approval: Handler<'approval'> = ctx => {
  const { request, sessionId } = ctx
  const p = request.params
  const description = p.description || 'dangerous command'

  rememberServerRequest(request)
  void receiveApprovalRequest(null, {
    // false only when a tirith warning forbids it; backend omits the field otherwise.
    allowPermanent: p.allow_permanent !== false,
    choices: p.choices,
    command: p.command,
    description,
    // The approval queue's own id — `approval.pending` / `approval.received` / `approval.respond` key on it.
    requestId: p.request_id ?? undefined,
    serverRequestId: request.id,
    sessionId: sessionId || null,
    smartDenied: p.smart_denied === true
  }).catch(() => undefined)
  markNeedsInput(ctx)

  if (!request.replayed) {
    dispatchNativeNotification({
      actions: [
        { id: 'approve', text: translateNow('notifications.native.approveAction') },
        { id: 'reject', text: translateNow('notifications.native.rejectAction') }
      ],
      body: p.command || description,
      kind: 'approval',
      sessionId: sessionId || null,
      title: translateNow('notifications.native.approvalTitle')
    })
  }
}

const sudo: Handler<'sudo'> = ctx => {
  rememberServerRequest(ctx.request)
  setSudoRequest({
    command: ctx.request.params.command,
    requestId: ctx.request.id,
    sessionId: ctx.sessionId || null
  })
  markNeedsInput(ctx)
  notifyInput(ctx, translateNow('notifications.native.inputBody'))
}

const secret: Handler<'secret'> = ctx => {
  const { env_var: envVar, prompt } = ctx.request.params

  rememberServerRequest(ctx.request)
  setSecretRequest({ envVar, prompt, requestId: ctx.request.id, sessionId: ctx.sessionId || null })
  markNeedsInput(ctx)
  notifyInput(ctx, prompt || envVar || translateNow('notifications.native.inputBody'))
}

const vaultCode: Handler<'vault.code'> = ctx => {
  const site = ctx.request.params.site ?? ''

  rememberServerRequest(ctx.request)
  setVaultCodeRequest({
    hint: ctx.request.params.hint ?? '',
    requestId: ctx.request.id,
    sessionId: ctx.sessionId || null,
    site
  })
  markNeedsInput(ctx)
  notifyInput(ctx, translateNow('prompts.vaultCodeTitle', site))
}

const vaultSaveLogin: Handler<'vault.save_login'> = ctx => {
  const { origin } = ctx.request.params
  const site = ctx.request.params.site || origin

  rememberServerRequest(ctx.request)
  setVaultSaveLoginRequest({ origin, requestId: ctx.request.id, sessionId: ctx.sessionId || null, site })
  markNeedsInput(ctx)
  notifyInput(ctx, translateNow('prompts.vaultSaveTitle', site))
}

const vaultUnlockPrompt: Handler<'vault.unlock_prompt'> = ctx => {
  const { backend } = ctx.request.params
  const displayName = ctx.request.params.display_name || backend

  rememberServerRequest(ctx.request)
  setVaultUnlockRequest({ backend, displayName, requestId: ctx.request.id, sessionId: ctx.sessionId || null })
  markNeedsInput(ctx)
  notifyInput(ctx, translateNow('prompts.vaultUnlockTitle', displayName))
}

// ── Desktop-surface bridges (answered immediately, no card) ─────────────────

type SurfaceRequest = ScopedServerRequest<'preview.act' | 'preview.read' | 'terminal.read' | 'tour' | 'window.read'>

interface RefusedAction {
  error: string
  success: false
}

type WindowBelow = Awaited<ReturnType<NonNullable<NonNullable<typeof window.hermesDesktop>['readWindowBelow']>>>

type SurfaceAnswer = null | PreviewActResult | PreviewReadResult | RefusedAction | TerminalReadResult | TourResult | WindowBelow

/** Answer a string-valued request with a JSON-encoded result ('' = nothing / unavailable). */
const answerValue = (request: SurfaceRequest, result: SurfaceAnswer) =>
  request.respond({ value: result ? JSON.stringify(result) : '' })

const refused = (error: string): RefusedAction => ({ error, success: false })

const failureText = (error: Error | string) => (error instanceof Error ? error.message : String(error))

const terminalRead: Handler<'terminal.read'> = ({ request }) => {
  // read_terminal tool: serialize the renderer's xterm buffer. Empty = no live pane.
  const { count, start } = request.params

  answerValue(request, readActiveTerminal({ count: count ?? undefined, start: start ?? undefined }))
}

const previewRead: Handler<'preview.read'> = ({ request }) => {
  // read_preview tool: the active preview tab's page text is async. Empty = nothing open.
  const { count, start } = request.params

  void readActivePreview({ count: count ?? undefined, start: start ?? undefined }).then(result =>
    answerValue(request, result)
  )
}

/** The wire operation as the preview engine takes it (nav verbs included; the engine routes those itself). */
const previewAction = (p: PreviewActParams): Omit<PreviewActAction, 'kind'> & { kind: PreviewActParams['action'] } => ({
  amount: p.amount ?? undefined,
  full: p.full ?? undefined,
  key: p.key ?? undefined,
  kind: p.action,
  max: p.max ?? undefined,
  ref: p.ref ?? undefined,
  selector: p.selector ?? undefined,
  submit: p.submit ?? undefined,
  text: p.text ?? undefined,
  to: p.to ?? undefined
})

const previewAct: Handler<'preview.act'> = ({ isActiveSession, request, sessionId }) => {
  // drive_preview tool: click/type/scroll/press inside the guest page. Active
  // session only: a background turn must never reach into the page the user is
  // working in (desktop AGENTS.md: offer, don't hijack). Every mounted window can
  // observe the same request; a scoped mismatch belongs to another window, so
  // answering here would race the owner — stay silent.
  if (sessionId && !isActiveSession) {
    return
  }

  if (!isActiveSession) {
    answerValue(request, refused('The in-app browser only takes actions in the session the user is looking at.'))

    return
  }

  void loadPreviewEngine()
    .then(run => run(previewAction(request.params)))
    .then(
      result => answerValue(request, result),
      error => answerValue(request, refused(failureText(error)))
    )
}

const windowRead: Handler<'window.read'> = ({ request }) => {
  // read_window_below tool: main owns native window enumeration. Empty =
  // unavailable (older shell without the handler, Wayland, …) — without an
  // answer the tool would stall its full 30s deadline.
  const read = window.hermesDesktop?.readWindowBelow

  void Promise.resolve(read ? read() : null).then(
    result => answerValue(request, result),
    () => answerValue(request, null)
  )
}

const tourStep = (step: WireTourStep): TourStep => ({
  selector: step.selector ?? undefined,
  side: step.side ?? undefined,
  text: step.text ?? undefined,
  title: step.title ?? undefined
})

const tourAction = (p: TourParams): TourAction => ({
  ...tourStep(p),
  kind: p.action,
  startAt: p.step_index ?? undefined,
  steps: p.steps?.map(tourStep)
})

const tour: Handler<'tour'> = ({ isActiveSession, request, sessionId }) => {
  // tour tool: one guided-tour action via driver.js, app DOM or preview guest
  // page. Active session only, same window-ownership rule as preview.act.
  if (sessionId && !isActiveSession) {
    return
  }

  if (!$toursEnabled.get()) {
    // Refused in words, not silently dropped: a no-op would leave the agent
    // narrating a spotlight the user can't see.
    answerValue(request, refused('The user has turned guided tours off.'))

    return
  }

  if (!isActiveSession) {
    answerValue(request, refused('Tours only run in the session the user is looking at.'))

    return
  }

  void import('@/lib/tour')
    .then(({ runTour }) => runTour(tourAction(request.params), request.params.surface ?? 'app'))
    .then(
      result => answerValue(request, result),
      error => answerValue(request, refused(failureText(error)))
    )
}

/** Method → handler. tsc holds this to every `ServerRequestMap` key. */
const HANDLERS: { [M in keyof ServerRequestMap]: Handler<M> } = {
  approval,
  clarify,
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

/** Dispatch one server→client request to its typed handler. */
export function handleServerRequest<M extends keyof ServerRequestMap>(
  request: ScopedServerRequest<M>,
  deps: ServerRequestContext['deps'],
  activeSessionId: null | string
): void {
  const sessionId = request.params.session_id

  HANDLERS[request.method]({
    deps,
    request,
    sessionId,
    isActiveSession: Boolean(sessionId) && sessionId === activeSessionId
  })
}

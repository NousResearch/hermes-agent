import { pendingClarifyToolPayload } from '@/app/session/hooks/use-session-actions/restore-pending-clarify'
import { translateNow } from '@/i18n'
import { restorePendingClarifyToolCall, settlePendingClarifyToolCall } from '@/lib/chat-messages'
import {
  $clarifyRequests,
  clearClarifyRequest,
  hasMalformedStructuredChoices,
  normalizeChoices,
  normalizeQuestions,
  setClarifyRequest,
  warnDroppedChoices
} from '@/store/clarify'
import { $gateway } from '@/store/gateway'
import { setMcpSetupRequest } from '@/store/mcp-setup'
import { dispatchNativeNotification } from '@/store/native-notifications'
import { receiveApprovalRequest, setSecretRequest, setSudoRequest } from '@/store/prompts'
import { requestScrollToBottom } from '@/store/thread-scroll'

import type { GatewayEventContext } from './types'

/** The blocking-input family: clarify / MCP setup consent / approval / sudo /
 *  secret requests. The Python side is blocked on the matching *.respond, so
 *  each of these must be parked per-session and surfaced. */
export function handleInputRequestEvent(ctx: GatewayEventContext): boolean {
  const { deps, event, payload, sessionId, occurredAt } = ctx
  const { activeSessionIdRef, sessionInterrupted, updateSessionState, upsertToolCall } = deps

  if (event.type === 'clarify.request') {
    // Surface the clarify tool's overlay. The Python side is blocked on
    // `clarify.respond`, so without this handler the agent would hang
    // forever (see tools/clarify_tool.py + tui_gateway/server.py:_block).
    //
    // Store the request for whichever session raised it — even a background
    // one. clarify.request is a one-shot event; if we dropped it for an
    // unfocused session, that session would block on `clarify.respond`
    // indefinitely and re-focusing it could never recover (the event is
    // gone). Parking it per-session lets the user answer once they switch
    // over; the inline ClarifyTool reads the active session's entry.
    if (sessionId && sessionInterrupted(sessionId)) {
      return true
    }

    const requestId = typeof payload?.request_id === 'string' ? payload.request_id : ''
    const question = typeof payload?.question === 'string' ? payload.question : ''
    const rawChoices = payload?.choices
    const choices = normalizeChoices(rawChoices)
    const multiSelect = payload?.multi_select === true
    const choicesMalformed = hasMalformedStructuredChoices(rawChoices, choices)
    // Batch (multi-question) clarify: `questions` replaces question/choices
    // on the wire. `answers` rides along only on reconnect replay, carrying
    // the per-question locks the server already accepted.
    const questions = normalizeQuestions(payload?.questions)

    const lockedAnswers =
      typeof payload?.answers === 'object' && payload?.answers !== null
        ? Object.fromEntries(
            Object.entries(payload.answers as Record<string, unknown>).filter(
              (entry): entry is [string, string] => typeof entry[1] === 'string'
            )
          )
        : undefined

    if (requestId && questions.length > 0) {
      const request = {
        choices: null,
        lockedAnswers,
        multiSelect: false,
        question: '',
        questions,
        receivedAt: Date.now() / 1000,
        requestId,
        sessionId: sessionId ?? null
      }

      setClarifyRequest(request)

      if (sessionId) {
        // A resumed/hydrated transcript may already contain this provider's
        // clarify call while carrying no live streamId. Re-arm that exact row
        // instead of letting the generic stream mutator append a second card.
        updateSessionState(sessionId, state => {
          const projection = restorePendingClarifyToolCall(
            state.messages,
            pendingClarifyToolPayload(request),
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

        if (sessionId === activeSessionIdRef.current) {
          requestScrollToBottom(sessionId)
        }
      }

      dispatchNativeNotification({
        body: questions.map(q => q.question).join(' · '),
        kind: 'input',
        sessionId,
        title: translateNow('notifications.native.inputTitle')
      })
    } else if (requestId && question) {
      if (choicesMalformed) {
        warnDroppedChoices('gateway', question, rawChoices)
      }

      const request = {
        requestId,
        question,
        choices: choices.length > 0 ? choices : null,
        multiSelect,
        choicesMalformed,
        receivedAt: Date.now() / 1000,
        sessionId: sessionId ?? null
      }

      setClarifyRequest(request)

      if (sessionId) {
        // Same provider-shape-aware repair as the batch path above. This keeps
        // the original hydrated row and provider tool id, while still seeding
        // a row when tool.start was genuinely missed.
        updateSessionState(sessionId, state => {
          const projection = restorePendingClarifyToolCall(
            state.messages,
            pendingClarifyToolPayload(request),
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

        if (sessionId === activeSessionIdRef.current) {
          requestScrollToBottom(sessionId)
        }
      }

      dispatchNativeNotification({
        body: question,
        kind: 'input',
        sessionId,
        title: translateNow('notifications.native.inputTitle')
      })
    }

    return true
  }

  if (event.type === 'clarify.expire') {
    if (!sessionId) {
      return true
    }

    const requestId = typeof payload?.request_id === 'string' ? payload.request_id : ''
    const request = $clarifyRequests.get()[sessionId]

    // Expiry is request-correlated: a delayed event from an older prompt must
    // not erase a newer clarify raised by the same session.
    if (!requestId || !request || request.requestId !== requestId) {
      return true
    }

    clearClarifyRequest(requestId, sessionId)
    updateSessionState(sessionId, state => {
      const projection = settlePendingClarifyToolCall(
        state.messages,
        pendingClarifyToolPayload(request),
        state.busy,
        occurredAt
      )

      return {
        ...state,
        messages: projection.messages,
        needsInput: false,
        streamId: state.busy ? (projection.streamId ?? state.streamId) : null
      }
    })

    return true
  }

  if (event.type === 'mcp.setup.request') {
    // setup_mcp tool (desktop GUI): the agent proposed an MCP server and
    // the Python side is blocked on mcp.setup.respond. Park the request
    // per-session (like clarify) and upsert a stable pending tool row so
    // the inline consent card has somewhere to render even when the
    // tool.start event was missed (stream reconnect / hydration race).
    const requestId = typeof payload?.request_id === 'string' ? payload.request_id : ''
    const server = typeof payload?.server === 'string' ? payload.server : ''
    const rawAction = typeof payload?.action === 'string' ? payload.action : 'install'
    const action = rawAction === 'enable' || rawAction === 'authorize' ? rawAction : 'install'
    const reason = typeof payload?.reason === 'string' ? payload.reason : ''

    if (requestId && server) {
      setMcpSetupRequest({ action, reason, requestId, server, sessionId: sessionId ?? null })

      if (sessionId) {
        upsertToolCall(
          sessionId,
          { args: { action, reason, server }, name: 'setup_mcp', tool_id: requestId },
          'running'
        )
        updateSessionState(sessionId, state => ({ ...state, needsInput: true }))
      }

      dispatchNativeNotification({
        body: reason || server,
        kind: 'input',
        sessionId,
        title: translateNow('notifications.native.inputTitle')
      })
    }

    return true
  }

  if (event.type === 'approval.request') {
    // Dangerous-command / execute_code approval. The Python side is blocked
    // in _await_gateway_decision() until approval.respond lands; without
    // this the agent stalls until its 5-min timeout and the tool is BLOCKED.
    // Park it per-session (like clarify) so a *background* profile's turn can
    // raise it and wait — the sidebar flags "needs input" and the inline bar
    // surfaces once the user focuses that chat.
    const command = typeof payload?.command === 'string' ? payload.command : ''
    const description = typeof payload?.description === 'string' ? payload.description : 'dangerous command'

    void receiveApprovalRequest($gateway.get(), {
      // false only when a tirith warning forbids it; backend omits the field otherwise.
      allowPermanent: payload?.allow_permanent !== false,
      choices: Array.isArray(payload?.choices)
        ? payload.choices.filter(choice => typeof choice === 'string')
        : undefined,
      command,
      description,
      requestId: typeof payload?.request_id === 'string' ? payload.request_id : undefined,
      sessionId: sessionId ?? null,
      smartDenied: payload?.smart_denied === true
    }).catch(() => undefined)

    if (sessionId) {
      updateSessionState(sessionId, state => ({ ...state, needsInput: true }))
    }

    dispatchNativeNotification({
      actions: [
        { id: 'approve', text: translateNow('notifications.native.approveAction') },
        { id: 'reject', text: translateNow('notifications.native.rejectAction') }
      ],
      body: command || description,
      kind: 'approval',
      sessionId,
      title: translateNow('notifications.native.approvalTitle')
    })

    return true
  }

  if (event.type === 'sudo.request') {
    // Sudo password capture (tools/terminal_tool.py). Blocked on
    // sudo.respond {request_id, password}.
    const requestId = typeof payload?.request_id === 'string' ? payload.request_id : ''

    if (requestId) {
      setSudoRequest({ requestId, sessionId: sessionId ?? null })

      if (sessionId) {
        updateSessionState(sessionId, state => ({ ...state, needsInput: true }))
      }

      dispatchNativeNotification({
        body: translateNow('notifications.native.inputBody'),
        kind: 'input',
        sessionId,
        title: translateNow('notifications.native.inputTitle')
      })
    }

    return true
  }

  if (event.type === 'secret.request') {
    // Skill credential capture (tools/skills_tool.py). Blocked on
    // secret.respond {request_id, value}.
    const requestId = typeof payload?.request_id === 'string' ? payload.request_id : ''

    if (requestId) {
      const envVar = typeof payload?.env_var === 'string' ? payload.env_var : ''
      const promptText = typeof payload?.prompt === 'string' ? payload.prompt : ''

      setSecretRequest({
        requestId,
        envVar,
        prompt: promptText,
        sessionId: sessionId ?? null
      })

      if (sessionId) {
        updateSessionState(sessionId, state => ({ ...state, needsInput: true }))
      }

      dispatchNativeNotification({
        body: promptText || envVar || translateNow('notifications.native.inputBody'),
        kind: 'input',
        sessionId,
        title: translateNow('notifications.native.inputTitle')
      })
    }

    return true
  }

  return false
}

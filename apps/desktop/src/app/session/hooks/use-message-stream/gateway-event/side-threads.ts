import { type ChatMessage, chatMessageText, textPart } from '@/lib/chat-messages'
import { coerceGatewayText } from '@/lib/chat-runtime'

import type { GatewayEventContext } from './types'

/** ``/btw`` side questions and ``/btw --explore`` side threads (agent/side_thread.py). */
export function handleSideThreadEvent(ctx: GatewayEventContext): boolean {
  const { event } = ctx

  if (event.type === 'btw.progress') {
    appendSideThreadStep(ctx)

    return true
  }

  if (event.type === 'btw.complete') {
    appendSideAnswer(ctx)

    return true
  }

  return false
}

/** One tool step of a /btw --explore side thread. Every step of a thread folds into ONE live row
 *  (`[btw <id>]` then a `┊ step` line each), so the side work reads as its own panel beside the
 *  main turn's stream. */
function appendSideThreadStep({ deps, occurredAt, payload, sessionId }: GatewayEventContext): void {
  const text = coerceGatewayText(payload?.text).trim()
  const taskId = String(payload?.task_id ?? '').trim()

  if (!text || !taskId || !sessionId) {
    return
  }

  const id = `btw-progress-${taskId}`

  deps.flushQueuedDeltas(sessionId)
  deps.updateSessionState(sessionId, state => {
    const existing = state.messages.find(message => message.id === id)

    const row: ChatMessage = {
      id,
      role: 'system',
      parts: [textPart(`${existing ? chatMessageText(existing) : `[btw ${taskId}]`}\n┊ ${text}`, occurredAt)],
      timestamp: existing?.timestamp ?? occurredAt
    }

    return {
      ...state,
      messages: existing
        ? state.messages.map(message => (message.id === id ? row : message))
        : [...state.messages, row]
    }
  })
}

/** prompt.btw answers a side question (or a side thread reports) and emits this on the
 *  originating session. Persistent transcript line, matching the TUI's `[btw "q"]` — without it
 *  Desktop only ever showed the acknowledgement (#99065). */
function appendSideAnswer({ deps, occurredAt, payload, sessionId }: GatewayEventContext): void {
  const text = coerceGatewayText(payload?.text).trim()

  if (!text || !sessionId) {
    return
  }

  const taskId = String(payload?.task_id ?? '').trim()
  const question = coerceGatewayText(payload?.question).trim()
  const header = `[btw${question ? ` "${question}"` : ''}${taskId ? ` (${taskId})` : ''}]`

  deps.flushQueuedDeltas(sessionId)
  deps.updateSessionState(sessionId, state => ({
    ...state,
    messages: [
      ...state.messages,
      {
        id: `btw-complete-${taskId || Date.now()}`,
        role: 'system',
        parts: [textPart(`${header}\n${text}`, occurredAt)],
        timestamp: occurredAt
      }
    ]
  }))
}

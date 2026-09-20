import { publishJarvisEvent } from '@/app/jarvis/store'
import type { JarvisEvent } from '@/app/jarvis/types'
import { coerceGatewayText } from '@/lib/chat-runtime'

import type { GatewayEventContext } from './types'

const JARVIS_VOICE_EVENT_TYPES = new Set(['voice.interrupted', 'voice.status', 'voice.transcript'])

const JARVIS_EVENT_TYPES: Record<string, JarvisEvent['type']> = {
  error: 'task.failed',
  'message.start': 'task.running',
  'tool.complete': 'tool.completed',
  'tool.start': 'tool.started',
  'voice.interrupted': 'voice.stopped',
  'voice.transcript': 'voice.stopped',
}

export function isJarvisVoiceGatewayEvent(eventType: string): boolean {
  return JARVIS_VOICE_EVENT_TYPES.has(eventType)
}

export function publishJarvisGatewayEvent(ctx: GatewayEventContext): void {
  const { deps, event, isActiveEvent, occurredAt, payload, sessionId, fromActiveSource } = ctx

  if (!sessionId || !isActiveEvent || !fromActiveSource()) {
    return
  }

  if (event.type === 'message.start' && deps.sessionStateByRuntimeIdRef.current.get(sessionId)?.interrupted) {
    return
  }

  const type = jarvisEventType(ctx)

  if (!type) {
    return
  }

  publishJarvisEvent({
    type,
    sessionId,
    at: occurredAt,
    label: typeof payload?.name === 'string' ? payload.name : undefined,
    detail: jarvisEventDetail(ctx),
    taskId: typeof payload?.task_id === 'string' ? payload.task_id : undefined,
    toolCallId: typeof payload?.tool_id === 'string' ? payload.tool_id : undefined,
  })
}

function jarvisEventType(ctx: GatewayEventContext): JarvisEvent['type'] | null {
  const { event, payload } = ctx

  if (event.type === 'approval.request') {
    return 'task.approval'
  }

  if (event.type === 'voice.status') {
    const state = (payload as { state?: unknown } | undefined)?.state

    return state === 'recording' ? 'voice.listening' : 'voice.stopped'
  }

  if (event.type === 'message.complete') {
    return payload?.status === 'error' ? 'task.failed' : 'task.verified'
  }

  return JARVIS_EVENT_TYPES[event.type] ?? null
}

function jarvisEventDetail(ctx: GatewayEventContext): string | undefined {
  const { event, payload } = ctx

  if (event.type === 'approval.request') {
    return typeof payload?.command === 'string' ? payload.command : undefined
  }

  if (event.type === 'error') {
    return typeof payload?.message === 'string' ? payload.message : undefined
  }

  if (event.type === 'message.complete' && payload?.status === 'error') {
    return (
      coerceGatewayText(payload.error).trim() ||
      coerceGatewayText(payload.text).trim() ||
      coerceGatewayText(payload.rendered).trim() ||
      undefined
    )
  }

  return undefined
}

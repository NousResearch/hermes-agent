import type { SessionControlUpdatePayload } from '@hermes/shared'

import { applySessionControlUpdate } from '@/store/session-control'

import type { GatewayEventContext } from './types'

export function handleControlEvent(ctx: GatewayEventContext): boolean {
  const { event, payload, sessionId } = ctx

  if (event.type !== 'session.control.update') {
    return false
  }

  if (!sessionId) {
    return true
  }

  const control = (payload as SessionControlUpdatePayload | undefined)?.control

  if (control) {
    applySessionControlUpdate(sessionId, control)
  }

  return true
}

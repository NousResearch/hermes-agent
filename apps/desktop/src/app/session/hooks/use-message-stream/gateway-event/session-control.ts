import { applySessionControlUpdate } from '@/store/session-control'

import type { GatewayEventContext } from './types'

export function handleControlEvent(ctx: GatewayEventContext): boolean {
  const { event, sessionId } = ctx

  if (event.type !== 'session.control.update') {
    return false
  }

  if (!sessionId) {
    return true
  }

  const control = event.payload?.control

  if (control) {
    applySessionControlUpdate(sessionId, control)
  }

  return true
}

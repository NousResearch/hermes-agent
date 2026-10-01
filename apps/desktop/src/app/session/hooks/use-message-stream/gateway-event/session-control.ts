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

  const control = payload && typeof payload === 'object' ? (payload as { control?: unknown }).control : undefined
  applySessionControlUpdate(sessionId, control)

  return true
}

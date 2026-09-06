import { beforeEach, expect, it } from 'vitest'

import type { AgentNoticePayload } from '@/store/agent-notices'
import { $notifications, clearNotifications } from '@/store/notifications'

import { handleStatusEvent } from './status'
import type { GatewayEventContext } from './types'

function context(type: string, connectionId: string, payload: AgentNoticePayload): GatewayEventContext {
  return {
    deps: {} as GatewayEventContext['deps'],
    event: { type, connectionId, profile: 'default', payload },
    explicitSid: 'same-runtime-id',
    sessionId: 'same-runtime-id',
    isActiveEvent: false,
    occurredAt: 1,
    payload: undefined,
    fromActiveSource: () => false,
    scheduleConfigRefresh: () => undefined
  }
}

beforeEach(clearNotifications)

it('routes background context warnings and recovery to their originating connection only', () => {
  const payload = {
    key: 'context-maintenance:same-profile:same-session',
    kind: 'sticky',
    level: 'warn',
    text: 'First connection'
  }

  handleStatusEvent(context('notification.show', 'first', payload))
  handleStatusEvent(context('notification.show', 'second', { ...payload, text: 'Second connection' }))
  expect($notifications.get()).toHaveLength(2)
  handleStatusEvent(context('notification.clear', 'first', { key: payload.key }))
  expect($notifications.get()).toHaveLength(1)
  expect($notifications.get()[0].message).toBe('Second connection')
})

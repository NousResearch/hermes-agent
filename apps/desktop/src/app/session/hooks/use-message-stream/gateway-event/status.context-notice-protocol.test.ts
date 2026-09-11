import { readFileSync } from 'node:fs'

import { beforeEach, expect, it } from 'vitest'

import type { AgentNoticePayload } from '@/store/agent-notices'
import { $notifications, clearNotifications } from '@/store/notifications'

import { handleStatusEvent } from './status'
import type { GatewayEventContext } from './types'

interface WireFrame {
  type: string
  seq: number
  payload: AgentNoticePayload & { state_revision?: number; state_key?: string }
}

interface Receipt {
  case: string
  frames: WireFrame[]
}

function context(frame: WireFrame, connectionId: string): GatewayEventContext {
  return {
    deps: {} as GatewayEventContext['deps'],
    event: { ...frame, connectionId, profile: 'default' },
    explicitSid: 'protocol-runtime',
    sessionId: 'protocol-runtime',
    isActiveEvent: false,
    occurredAt: 1,
    payload: undefined,
    fromActiveSource: () => false,
    scheduleConfigRefresh: () => undefined
  }
}

// CI contract fixture. A verification run can instead consume the exact JSON
// frames produced by test_context_notice_protocol.py's real SQLite/compute wire.
const key = 'context-maintenance:protocol-fixture:conversation'
const warning = { key, text: 'Summary route unavailable', kind: 'sticky', level: 'warn', state_key: key }
const recovery = {
  key: `${key}:recovered`,
  text: 'Context maintenance has recovered.',
  kind: 'ttl',
  level: 'success',
  ttl_ms: 5000,
  state_key: key,
  state_revision: 3
}
const fixture: Receipt = {
  case: 'contract fixture',
  frames: [
    { type: 'notification.show', seq: 1, payload: { ...warning, state_revision: 2 } },
    { type: 'notification.clear', seq: 2, payload: { key, state_key: key, state_revision: 3 } },
    { type: 'notification.show', seq: 3, payload: recovery },
    { type: 'notification.show', seq: 4, payload: { ...warning, state_revision: 2 } },
    { type: 'notification.clear', seq: 5, payload: { key, state_key: key, state_revision: 4 } },
    { type: 'notification.show', seq: 6, payload: { ...warning, state_revision: 5 } },
    { type: 'notification.show', seq: 7, payload: recovery }
  ]
}
const receipts: Receipt[] = process.env.CONTEXT_NOTICE_WIRE_RECEIPT
  ? JSON.parse(readFileSync(process.env.CONTEXT_NOTICE_WIRE_RECEIPT, 'utf8'))
  : [fixture]

beforeEach(clearNotifications)

it.each(receipts)('$case: stale snapshot and recovery TTL cannot override newer durable state', receipt => {
  const deliver = (frame: WireFrame) => handleStatusEvent(context(frame, receipt.case))
  receipt.frames.slice(0, 4).forEach(deliver)
  expect($notifications.get()).toHaveLength(1)
  expect($notifications.get()[0].kind).toBe('success')
  deliver(receipt.frames[4])
  expect($notifications.get()).toHaveLength(0)
  receipt.frames.slice(5).forEach(deliver)
  expect($notifications.get()).toHaveLength(1)
  expect($notifications.get()[0].kind).toBe('warning')
})

it('keeps revision watermarks connection-scoped and preserves legacy notice clients', () => {
  handleStatusEvent(context(fixture.frames[5], 'connection-a'))
  handleStatusEvent(context(fixture.frames[0], 'connection-b'))
  handleStatusEvent(context(fixture.frames[1], 'connection-b'))
  expect($notifications.get()).toHaveLength(1)
  const legacy = { type: 'notification.show', seq: 1, payload: { key: 'legacy-warning', text: 'Still supported' } }
  handleStatusEvent(context(legacy, 'connection-a'))
  expect($notifications.get()).toHaveLength(2)
  handleStatusEvent(context({ ...legacy, type: 'notification.clear' }, 'connection-a'))
  expect($notifications.get()).toHaveLength(1)
})

import { describe, expect, it } from 'vitest'

import {
  applyHostedPage, hostedHolds, hostedMembers, hostedPendingActions, hostedRoomKey, hostedTranscript,
  isHostedRoomKey, parseHostedCapabilities, parseHostedRoom
} from './hosted-room-protocol'
import type { HostedEvent, HostedReplay, HostedRoomSummary } from './hosted-room-protocol'

// Synthetic v2 wire fixtures, not agent output or evidence of a live gateway.
const identity = { connectionId: 'mock-connection', authorityGatewayId: 'mock-authority', roomId: 'mock-room' }

const room: HostedRoomSummary = {
  room_id: identity.roomId, authority_gateway_id: identity.authorityGatewayId,
  authority_epoch: 1, latest_seq: 3, name: 'Mock room',
  members: [{ member_id: 'mock-member', profile: 'helper', handle: 'helper' }]
}

const capabilities = {
  protocol_version: 2, authority_gateway_id: identity.authorityGatewayId,
  methods: ['groups.list', 'groups.state', 'groups.send', 'groups.log', 'groups.stop'],
  features: ['room_identity', 'monotonic_log', 'idempotent_send', 'typed_events'],
  max_log_limit: 500, driver: true, persistent_process: true
}

function event(seq: number, patch: Partial<HostedEvent> = {}): HostedEvent {
  return {
    room_id: room.room_id, seq, event_id: `mock-event-${seq}`, kind: 'message.user',
    actor: { kind: 'user', id: 'mock-user' }, payload: { text: 'Mock text', thread_id: 'mock-thread' },
    created_at: 1, authority_epoch: null, ...patch
  }
}

function page(events: HostedEvent[], latest = events.at(-1)?.seq ?? 0, cursor = events.at(-1)?.seq ?? 0) {
  return {
    events, latest_seq: latest, cursor, has_more: cursor < latest,
    authority: { gateway_id: room.authority_gateway_id, epoch: room.authority_epoch }
  }
}

it('keys identity by connection, authority and room, not display name or ambiguous delimiters', () => {
  const key = hostedRoomKey(identity)
  expect(isHostedRoomKey(key)).toBe(true)
  expect(isHostedRoomKey(room.name)).toBe(false)

  for (const field of ['connectionId', 'authorityGatewayId', 'roomId'] as const) {
    expect(hostedRoomKey({ ...identity, [field]: `${identity[field]}:other` })).not.toBe(key)
  }

  expect(hostedRoomKey({ ...identity, connectionId: 'a:b', authorityGatewayId: 'c' }))
    .not.toBe(hostedRoomKey({ ...identity, connectionId: 'a', authorityGatewayId: 'b:c' }))
})

describe('capability negotiation', () => {
  it('accepts v2, caps pages, and treats worker/lifetime flags as strict booleans', () => {
    expect(parseHostedCapabilities(capabilities)).toEqual({
      authorityGatewayId: identity.authorityGatewayId, logLimit: 100, driver: true, persistentProcess: true,
      persistentHolds: false, attachments: false
    })
    expect(parseHostedCapabilities({ ...capabilities, max_log_limit: 7, driver: 'true', persistent_process: 1 }))
      .toMatchObject({ logLimit: 7, driver: false, persistentProcess: false })
  })
  it.each([
    ['old version', { protocol_version: 1 }],
    ['missing authority', { authority_gateway_id: '' }],
    ['zero limit', { max_log_limit: 0 }],
    ['fractional limit', { max_log_limit: 1.5 }],
    ['string limit', { max_log_limit: '100' }],
    ...capabilities.methods.map((method): [string, Record<string, unknown>] => [`missing ${method}`, { methods: capabilities.methods.filter(m => m !== method) }]),
    ...capabilities.features.map((feature): [string, Record<string, unknown>] => [`missing ${feature}`, { features: capabilities.features.filter(f => f !== feature) }])
  ])('rejects %s without a legacy negotiation path', (_name, patch) => {
    expect(() => parseHostedCapabilities({ ...capabilities, ...patch as object })).toThrow()
  })
  it('reads durable holds as an optional feature, never as a negotiation requirement', () => {
    const holding = { ...capabilities, features: [...capabilities.features, 'persistent_member_holds'] }
    expect(parseHostedCapabilities(holding).persistentHolds).toBe(true)
    // A gateway without the feature still negotiates: it just cannot promise a pause sticks.
    expect(parseHostedCapabilities(capabilities).persistentHolds).toBe(false)
  })
})

describe('backend-owned pending actions', () => {
  const approval = {
    kind: 'approval', task_id: 'task-a', member_id: 'member-a', execution_generation: 1, request_id: 'request-a',
    approval: { command: 'printf "<review me>"', reason: 'Shell command', choices: ['once', 'deny'] }
  }

  it('preserves the native command and reason so approval is informed', () => {
    expect(hostedPendingActions({ pending_actions: [approval] })).toEqual([{
      kind: 'approval', taskId: 'task-a', memberId: 'member-a', executionGeneration: 1,
      requestId: 'request-a', command: 'printf "<review me>"', reason: 'Shell command', choices: ['once', 'deny']
    }])
  })

  it.each([undefined, '', 42])('allows only denial when the command cannot be inspected (%s)', command => {
    expect(hostedPendingActions({ pending_actions: [{ ...approval, approval: { ...approval.approval, command } }] }))
      .toMatchObject([{ choices: ['deny'], command: '' }])
  })

  it('retains retry identity but ignores approval without an exact request and generation', () => {
    expect(hostedPendingActions({ pending_actions: [
      { kind: 'retry', task_id: 'task-b' },
      { ...approval, request_id: undefined },
      { ...approval, execution_generation: true }
    ] })).toEqual([{ kind: 'retry', taskId: 'task-b' }])
  })
})

describe('backend-owned member holds', () => {
  it('labels held members from the gateway status, preferring a display name', () => {
    expect(hostedHolds({
      running: true,
      holds: [
        { member_id: 'mock-member', handle: 'helper', display_name: 'Helper Bot', held_at_seq: 4 },
        { member_id: 'mock-second', handle: 'second', display_name: '  ' }
      ]
    })).toEqual([
      { memberId: 'mock-member', handle: 'helper', label: 'Helper Bot' },
      { memberId: 'mock-second', handle: 'second', label: 'second' }
    ])
  })
  it.each([
    ['no status', undefined],
    ['no holds field', { running: true }],
    ['holds is not a list', { holds: {} }],
    ['rows without a member id', { holds: [{ handle: 'helper' }, 'helper', null] }]
  ])('reports nothing held for %s rather than inventing one', (_name, status) => {
    expect(hostedHolds(status)).toEqual([])
  })
  it('falls back to the member id when the gateway sends no handle', () => {
    expect(hostedHolds({ holds: [{ member_id: 'mock-member' }] }))
      .toEqual([{ memberId: 'mock-member', handle: 'mock-member', label: 'mock-member' }])
  })
})

describe('authority and replay', () => {
  it.each([
    { room_id: 'other-room' }, { authority_gateway_id: 'other-authority' }
  ])('fails closed when room identity changes: %j', patch => {
    expect(() => parseHostedRoom({ ...room, ...patch }, identity)).toThrow(/authority/)
  })

  it('counts control events, follows the validated cursor, and never renders controls as chat', () => {
    const first = applyHostedPage({ cursor: 0, events: [] }, page([event(1, { kind: 'room.created', payload: {} })], 3), room)
    expect(first.cursor).toBe(1)
    expect(first.hasMore).toBe(true)

    const second = applyHostedPage(first, page([
      event(2), event(3, { kind: 'message.member', payload: { text: 'Mock reply', thread_id: 'mock-thread', member_id: 'mock-member' } })
    ]), room)

    expect(second.cursor).toBe(3)
    expect(second.hasMore).toBe(false)
    expect(hostedTranscript(second).log.map(row => [row.id, row.from.kind, row.from.name])).toEqual([
      ['mock-event-2', 'user', 'mock-user'], ['mock-event-3', 'member', 'mock-member']
    ])
    expect(hostedMembers(room)[0].name).toBe('mock-member')
  })

  it('preserves the event array on empty no-op pages', () => {
    const previous = { cursor: 1, events: [event(1)] }
    expect(applyHostedPage(previous, page([], 1, 1), { ...room, latest_seq: 1 }).events).toBe(previous.events)
  })

  it.each(['message.user', 'message.member'])('validates %s manifests before admitting a renderable page', kind => {
    const attachment = { attachment_id: `att_${'a'.repeat(32)}`, kind: 'file', name: 'mock.txt', mime: 'text/plain', size: 4 }
    const previous = { cursor: 1, events: [event(1)] }
    const transcript = hostedTranscript(previous)
    const payload = { text: 'Text with a file', thread_id: 'mock-thread', member_id: 'mock-member', attachments: [attachment] }

    for (const attachments of [
      {}, [{ ...attachment, attachment_id: 'invalid' }], [{ ...attachment, kind: 'unknown' }],
      [{ ...attachment, name: '' }], [{ ...attachment, mime: '' }], [{ ...attachment, size: 1.5 }]
    ]) {
      const bad = event(3, { kind, payload: { ...payload, attachments } })
      expect(() => hostedTranscript({ cursor: 3, events: [bad] })).toThrow()
      expect(() => applyHostedPage(previous, page([event(2), bad]), room)).toThrow()
      expect(previous.cursor).toBe(1)
      expect(hostedTranscript(previous)).toEqual(transcript)
    }

    const accepted = applyHostedPage(previous, page([event(2), event(3, { kind, payload })]), room)
    expect(hostedTranscript(accepted).log.at(-1)?.images).toEqual([{
      attachmentId: attachment.attachment_id, kind: 'file', name: attachment.name,
      mime: attachment.mime, size: attachment.size, data: ''
    }])
  })

  it.each([
    ['gap', page([event(2), event(4)])],
    ['reordered', page([event(3), event(2)])],
    ['wrong room', page([event(2, { room_id: 'wrong' })])],
    ['duplicate across pages', page([event(2, { event_id: 'mock-event-1' })])],
    ['duplicate within page', page([event(2), event(3, { event_id: 'mock-event-2' })])],
    ['bad timestamp', page([event(2, { created_at: Number.NaN })])],
    ['missing message thread', page([event(2, { payload: { text: 'Mock text' } })])],
    ['missing member id', page([event(2, { kind: 'message.member' })])],
    ['future epoch', page([event(2, { authority_epoch: 2 })])],
    ['invalid actor', page([event(2, { actor: { kind: 'user', id: '' } })])],
    ['wrong authority', { ...page([event(2)]), authority: { gateway_id: 'wrong', epoch: 1 } }],
    ['wrong page epoch', { ...page([event(2)]), authority: { gateway_id: identity.authorityGatewayId, epoch: 2 } }],
    ['jumped cursor', { ...page([event(2)]), cursor: 3 }],
    ['latest behind cursor', { ...page([event(2)]), latest_seq: 1 }],
    ['false has_more', { ...page([event(2)], 3), has_more: false }],
    ['empty nonterminal page', page([], 3, 1)],
    ['malformed events', { ...page([]), events: {} }]
  ])('rejects %s atomically without advancing the old cursor', (_name, raw) => {
    const previous: HostedReplay = { cursor: 1, events: [event(1)] }
    const before = structuredClone(previous)
    expect(() => applyHostedPage(previous, raw, room)).toThrow()
    expect(previous).toEqual(before)
  })

  it('does not accept a terminal page behind the already observed room high-water mark', () => {
    // State proves seq 3 exists. A purported complete seq-2 log cannot be current.
    expect(() => applyHostedPage({ cursor: 0, events: [] }, page([event(1), event(2)]), room)).toThrow()
  })
})

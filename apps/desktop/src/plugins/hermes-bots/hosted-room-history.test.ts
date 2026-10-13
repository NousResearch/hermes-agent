import { expect, it } from 'vitest'

import { applyHostedPage, hostedTranscript } from './hosted-room-protocol'
import type { HostedEvent, HostedRoomSummary } from './hosted-room-protocol'

const sources = ['desktop', 'telegram'].map(surface => ({
  surface, room_name: `Old ${surface}`, sessions: { bot: { id: `${surface}-stored`, title: `Group: ${surface}` } },
  records: [{ id: 'collision', at: 1234, from: { kind: 'member', name: 'bot' }, text: `verbatim ${surface}`, thread: 'same' }]
}))

const event: HostedEvent = {
  room_id: 'room', seq: 1, event_id: 'system:legacy-history-adoption', kind: 'room.created',
  actor: { kind: 'system', id: 'legacy-history-adoption' }, authority_epoch: 1, created_at: 999,
  payload: { legacy_history: { version: 1, sources } }
}

it('projects distinct inert archives without rewriting their records or provenance', () => {
  const before = structuredClone(event)

  const room: HostedRoomSummary = {
    room_id: 'room', name: 'Shared', members: [], authority_gateway_id: 'host', authority_epoch: 1, latest_seq: 1
  }

  const replay = applyHostedPage({ cursor: 0, events: [] }, {
    authority: { gateway_id: 'host', epoch: 1 }, events: [event], cursor: 1, latest_seq: 1, has_more: false
  }, room)

  const log = hostedTranscript(replay).log

  expect(log).toHaveLength(2)
  expect(log[0].id).not.toBe(log[1].id)
  expect(log[0].thread).not.toBe(log[1].thread)
  expect(log[0].at).toBe(1234)
  expect(log[0].text).toBe('[Imported Desktop history, not a native turn]\n\nverbatim desktop')
  expect(log[1].text).toBe('[Imported Telegram history, not a native turn]\n\nverbatim telegram')
  expect(event).toEqual(before)
})

it('rejects a missing source and leaves non-adoption room events unchanged', () => {
  const bad: HostedEvent = {
    ...event, payload: { legacy_history: { version: 1, sources: sources.slice(0, 1) } }
  }

  expect(() => hostedTranscript({ cursor: 1, events: [bad] })).toThrow('Both legacy')
  const previous = { cursor: 1, events: [{ ...event, event_id: 'good-created', payload: {} }] }
  const transcript = hostedTranscript(previous)
  expect(() => applyHostedPage(previous, {
    authority: { gateway_id: 'host', epoch: 1 }, events: [{ ...bad, seq: 2 }], cursor: 2, latest_seq: 2, has_more: false
  }, {
    room_id: 'room', name: 'Shared', members: [], authority_gateway_id: 'host', authority_epoch: 1, latest_seq: 2
  })).toThrow('Both legacy')
  expect(previous.cursor).toBe(1)
  expect(hostedTranscript(previous)).toEqual(transcript)
  expect(hostedTranscript({ cursor: 1, events: [{ ...event, payload: {} }] }).log).toEqual([])
})

it('maps frozen source threads to stable bounded native reply IDs without changing the archive', () => {
  const originals = ['same', 'other', 'same', '\u4e2d/'.repeat(5000), { id: ['arbitrary', 42] },
    { id: ['arbitrary', 42] }, null, undefined, 'legacy', 42, '42']

  const archive = {
    ...event, payload: { legacy_history: { version: 1, sources: sources.map(source => ({
      ...source, records: originals.map((thread, index) => ({ ...source.records[0], id: `record-${index}`, thread }))
    })) } }
  }

  const before = structuredClone(archive)
  const later = { ...archive, seq: 2, event_id: 'second-adoption' }

  const room: HostedRoomSummary = {
    room_id: 'room', name: 'Shared', members: [], authority_gateway_id: 'host', authority_epoch: 1, latest_seq: 2
  }

  const replay = applyHostedPage({ cursor: 0, events: [] }, {
    authority: { gateway_id: 'host', epoch: 1 }, events: [archive, later], cursor: 2, latest_seq: 2, has_more: false
  }, room)

  const projected = hostedTranscript(replay).log
  const ordinals = [0, 1, 0, 2, 3, 3, 4, 4, 4, 5, 6]
  expect(projected.map(row => row.thread)).toEqual([1, 2].flatMap(seq => sources.flatMap(source =>
    ordinals.map(ordinal => `legacy-history:${seq}:${source.surface}:${ordinal}`))))
  expect(hostedTranscript(replay).log).toEqual(projected)
  expect(archive).toEqual(before)

  for (const row of hostedTranscript({ cursor: Number.MAX_SAFE_INTEGER,
    events: [{ ...archive, seq: Number.MAX_SAFE_INTEGER }] }).log) {
    expect(row.thread).toMatch(/^[A-Za-z0-9][A-Za-z0-9._:-]*$/)
    expect(row.thread!.length).toBeLessThan(128)
  }

  const reply: HostedEvent = { ...event, seq: 3, event_id: 'native-reply', kind: 'message.user',
    actor: { kind: 'user', id: 'desktop' }, payload: { text: 'Continue this discussion', thread_id: projected[0].thread } }

  const continued = applyHostedPage(replay, {
    authority: { gateway_id: 'host', epoch: 1 }, events: [reply], cursor: 3, latest_seq: 3, has_more: false
  }, { ...room, latest_seq: 3 })

  expect(hostedTranscript(continued).log.filter(row => row.thread === projected[0].thread).map(row => row.id))
    .toEqual([projected[0].id, projected[2].id, reply.event_id])
})

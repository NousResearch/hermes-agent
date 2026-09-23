import { describe, expect, it } from 'vitest'

import { type A2aEvent, a2aEventsForSide, a2aThreadState, type A2aTranscriptRow, mergeA2aThread } from './a2a-thread'

const at = (seconds: number) => 1_758_300_000 + seconds

const dm = (from: string, body: string, seconds: number): A2aTranscriptRow => ({
  content: `Message from 🤖 ${from} (@${from.replace(/ /g, '-')}): ${body}`,
  role: 'user',
  timestamp: at(seconds)
})

const says = (text: string, seconds: number): A2aTranscriptRow => ({ content: text, role: 'assistant', timestamp: at(seconds) })

const sends = (target: string, message: string, seconds: number): A2aTranscriptRow => ({
  content: '',
  role: 'assistant',
  timestamp: at(seconds),
  tool_calls: [{ function: { arguments: JSON.stringify({ message, target }), name: 'message_agent' }, id: 'call-1', type: 'function' }]
})

describe('a2aEventsForSide', () => {
  it('reads a received DM as an event, prefix stripped', () => {
    const events = a2aEventsForSide([dm('hotel dev', 'SDK gap report attached', 10)], 'platform-engineer', 'hotel-dev')

    expect(events).toEqual([
      { from: 'hotel-dev', kind: 'dm', text: 'SDK gap report attached', to: 'platform-engineer', ts: at(10) * 1000 }
    ])
  })

  it('pairs the DM with the LAST assistant text of the turn it started', () => {
    const events = a2aEventsForSide(
      [dm('hotel dev', 'gap report', 10), says('Reading the report…', 12), says('Confirmed — shipping the fix.', 20)],
      'platform-engineer',
      'hotel-dev'
    )

    expect(events.filter(event => event.kind === 'reply')).toEqual([
      { from: 'platform-engineer', kind: 'reply', text: 'Confirmed — shipping the fix.', to: 'hotel-dev', ts: at(20) * 1000 }
    ])
  })

  it('never attributes a reply across an interrupting human message', () => {
    const events = a2aEventsForSide(
      [dm('hotel dev', 'gap report', 10), { content: 'stop everything', role: 'user', timestamp: at(11) }, says('Done.', 12)],
      'platform-engineer',
      'hotel-dev'
    )

    expect(events.map(event => event.kind)).toEqual(['dm'])
  })

  it('reads an outbound message_agent call as a sent event', () => {
    const events = a2aEventsForSide([sends('@hotel-dev', 'GO for Phase 1.', 30)], 'platform-engineer', 'hotel-dev')

    expect(events).toEqual([{ from: 'platform-engineer', kind: 'sent', text: 'GO for Phase 1.', to: 'hotel-dev', ts: at(30) * 1000 }])
  })

  it('ignores a send addressed to a third bot', () => {
    const events = a2aEventsForSide([sends('sts-designer', 'unrelated', 30)], 'platform-engineer', 'hotel-dev')

    expect(events).toEqual([])
  })
})

describe('mergeA2aThread', () => {
  it('drops the sender-side duplicate of a delivery the receiver recorded', () => {
    const sent = a2aEventsForSide([sends('hotel-dev', 'GO for Phase 1.', 30)], 'platform-engineer', 'hotel-dev')
    const received = a2aEventsForSide([dm('platform-engineer', 'GO for Phase 1.', 32)], 'hotel-dev', 'platform-engineer')

    const thread = mergeA2aThread(sent, received)

    expect(thread).toHaveLength(1)
    expect(thread[0].kind).toBe('dm')
    expect(thread[0].to).toBe('hotel-dev')
  })

  it('pairs each delivery with the send it followed, not the first match in the window', () => {
    // "OK" twice, only the second one delivered: the window's first match is the
    // FIRST send, so proximity alone consumed a message that really was sent.
    const sent = a2aEventsForSide(
      [sends('hotel-dev', 'OK', 30), sends('hotel-dev', 'OK', 600)],
      'platform-engineer',
      'hotel-dev'
    )
    const received = a2aEventsForSide([dm('platform-engineer', 'OK', 610)], 'hotel-dev', 'platform-engineer')

    const thread = mergeA2aThread(sent, received)
    const remainingSends = thread.filter(event => event.kind === 'sent')

    expect(remainingSends).toHaveLength(1)
    expect(thread.filter(event => event.kind === 'dm')).toHaveLength(1)
    // The survivor is the EARLIER send (the later one became the delivery).
    expect(remainingSends[0].ts).toBeLessThan(thread.find(event => event.kind === 'dm')!.ts)
  })

  it('keeps both copies of a repeated identical message that was delivered twice', () => {
    const sent = a2aEventsForSide(
      [sends('hotel-dev', 'OK', 30), sends('hotel-dev', 'OK', 600)],
      'platform-engineer',
      'hotel-dev'
    )
    const received = a2aEventsForSide(
      [dm('platform-engineer', 'OK', 32), dm('platform-engineer', 'OK', 610)],
      'hotel-dev',
      'platform-engineer'
    )

    expect(mergeA2aThread(sent, received).map(event => event.kind)).toEqual(['dm', 'dm'])
  })

  it('orders both sides into one timeline', () => {
    const platform = a2aEventsForSide(
      [dm('hotel dev', 'gap report', 10), says('Confirmed — shipping.', 20), sends('hotel-dev', 'Ready for verification.', 40)],
      'platform-engineer',
      'hotel-dev'
    )

    const hotel = a2aEventsForSide([dm('platform-engineer', 'Ready for verification.', 42)], 'hotel-dev', 'platform-engineer')

    const thread = mergeA2aThread(platform, hotel)

    expect(thread.map(event => [event.from, event.kind])).toEqual([
      ['hotel-dev', 'dm'],
      ['platform-engineer', 'reply'],
      ['platform-engineer', 'dm']
    ])
  })
})

describe('a2aThreadState', () => {
  const event = (kind: A2aEvent['kind'], ts: number): A2aEvent => ({ from: 'a', kind, text: 'x', to: 'b', ts })

  it('is settled with nothing to say about an empty thread', () => {
    expect(a2aThreadState([])).toEqual({ lastTs: 0, state: 'settled', waitingOn: null })
  })

  it('reads a fresh event as active', () => {
    const now = at(100) * 1000
    expect(a2aThreadState([event('reply', now - 5_000)], now)).toEqual({ lastTs: now - 5_000, state: 'active', waitingOn: null })
  })

  it('names who still owes a reply once the last word is an unanswered DM', () => {
    const now = at(10_000) * 1000
    const stale = a2aThreadState([event('dm', now - 60 * 60 * 1000)], now)

    expect(stale).toEqual({ lastTs: now - 60 * 60 * 1000, state: 'awaiting-reply', waitingOn: 'b' })
  })

  it('reads a stale reply as settled', () => {
    const now = at(10_000) * 1000
    expect(a2aThreadState([event('dm', now - 7_200_000), event('reply', now - 3_600_000)], now)).toEqual({
      lastTs: now - 3_600_000,
      state: 'settled',
      waitingOn: null
    })
  })
})

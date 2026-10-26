import { describe, expect, it } from 'vitest'

import { senderForSession, type TeammateDelivery, visibleDeliveries } from './teammate-waits'
import type { RosterRow } from './types'

const bot = (name: string, id: string, resolved = id): RosterRow =>
  ({ name, canonical_session: { id, resolved_id: resolved } }) as RosterRow

const row = (id: string, state: TeammateDelivery['state']): TeammateDelivery => ({
  delivery_id: id,
  target: 'scribe',
  target_handle: 'scribe',
  state,
  since: 0,
  preview: ''
})

describe('teammate-wait strip', () => {
  it('belongs only to the composer of a bot’s own canonical chat, tip or durable id', () => {
    const roster = [bot('alice', 'a-root', 'a-tip'), bot('scribe', 's-root')]

    expect(senderForSession(roster, ['a-tip', null])?.name).toBe('alice')
    expect(senderForSession(roster, [null, 's-root'])?.name).toBe('scribe')
    expect(senderForSession(roster, ['some-other-chat', null])).toBeNull()
    expect(senderForSession([{ ...bot('alice', 'a-root'), ghost: true }], ['a-root'])).toBeNull()
  })

  it('only a failure can be dismissed; a live wait never hides', () => {
    const rows = [row('1', 'failed'), row('2', 'waiting_on_you'), row('3', 'failed')]

    expect(visibleDeliveries(rows, { '1': true, '2': true }).map(r => r.delivery_id)).toEqual(['2', '3'])
  })
})

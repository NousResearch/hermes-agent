import { describe, expect, it } from 'vitest'

import { TEAMMATE_WAIT_LOCALES } from './teammate-wait-locales'
import {
  failureReasonText,
  senderForSession,
  type TeammateDelivery,
  teammateFor,
  visibleDeliveries
} from './teammate-waits'
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

  it('a failure reads in plain words in every locale, never as the raw reason code', () => {
    for (const messages of Object.values(TEAMMATE_WAIT_LOCALES)) {
      for (const code of ['provider_auth_or_access', 'runtime_offline', 'some_future_code', 'constructor', null]) {
        expect(failureReasonText(code, null, messages.reasons)).not.toMatch(/_/)
      }

      expect(failureReasonText('some_future_code', null, messages.reasons)).toBe(messages.reasons.unknown)
      // An unclassified failure shows what the teammate's owner actually saw.
      expect(failureReasonText('unknown', 'Connection error.', messages.reasons)).toBe('Connection error.')
      expect(failureReasonText('provider_auth_or_access', 'HTTP 401', messages.reasons)).toBe(
        messages.reasons.provider_auth_or_access
      )
    }
  })

  it('a teammate the roster has not caught up with is never the sender in disguise', () => {
    const alice = { ...bot('alice', 'a-root'), connectionId: 'box' } as RosterRow
    const scribe = { ...bot('scribe', 's-root'), connectionId: 'box' } as RosterRow
    const delivery = { ...row('1', 'queued'), target: 'scribe' }

    expect(teammateFor([alice, scribe], alice, delivery)).toBe(scribe)
    expect(teammateFor([alice], alice, { ...delivery, target: 'newcomer' })).toBeNull()
  })
})

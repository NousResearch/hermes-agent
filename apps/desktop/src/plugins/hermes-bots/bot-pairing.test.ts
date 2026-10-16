import { describe, expect, it } from 'vitest'

import { pairingInstruction, pairingProblem, pairableBots } from './bot-pairing'
import type { RosterRow } from './types'

const row = (over: Partial<RosterRow> & { name: string }): RosterRow =>
  ({ ...over }) as RosterRow

describe('pairingProblem', () => {
  it('requires both bots picked', () => {
    const a = row({ name: 'atlas' })
    expect(pairingProblem(a, null)).toBe('Pick both bots.')
    expect(pairingProblem(null, a)).toBe('Pick both bots.')
  })

  it('rejects pairing a bot with itself', () => {
    expect(pairingProblem(row({ name: 'atlas' }), row({ name: 'atlas' }))).toBe('Pick two different bots.')
  })

  it('rejects ghosts and unreachable sources', () => {
    expect(pairingProblem(row({ name: 'a' }), row({ name: 'b', ghost: true }))).toContain('Offline')
    expect(
      pairingProblem(row({ name: 'a' }), row({ name: 'b', sourceReachable: false }))
    ).toContain('unreachable')
  })

  it('accepts a valid distinct pair', () => {
    expect(pairingProblem(row({ name: 'atlas' }), row({ name: 'dev' }))).toBeNull()
  })
})

describe('pairableBots', () => {
  it('filters ghosts and unreachable rows, keeps the rest', () => {
    const roster = [
      row({ name: 'a' }),
      row({ name: 'ghosted', ghost: true }),
      row({ name: 'offline', sourceReachable: false }),
      null as unknown as RosterRow,
      row({ name: 'b' })
    ]

    expect(pairableBots(roster).map(bot => bot.name)).toEqual(['a', 'b'])
  })
})

describe('pairingInstruction', () => {
  it('names the responder and directs the message_agent call', () => {
    const text = pairingInstruction({
      initiator: row({ name: 'atlas' }),
      responder: row({ name: 'dev' })
    })

    expect(text).toContain('@dev')
    expect(text).toContain('message_agent')
    // Fire-and-forget contract: the bot must not sit waiting on the user.
    expect(text).toContain('Do not wait for the user after messaging')
  })

  it('weaves the optional joint task in', () => {
    const text = pairingInstruction({
      initiator: row({ name: 'atlas' }),
      responder: row({ name: 'dev' }),
      task: 'planejar o deploy da v2'
    })

    expect(text).toContain('planejar o deploy da v2')
  })

  it('falls back to open-ended introductions without a task', () => {
    const text = pairingInstruction({
      initiator: row({ name: 'atlas' }),
      responder: row({ name: 'dev' })
    })

    expect(text).toContain('Introduce yourselves')
  })
})

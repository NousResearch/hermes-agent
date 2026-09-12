import { describe, expect, it } from 'vitest'

import { resolveGroupResponders } from './group-rounds'
import type { GroupMember, GroupMessage } from './types'

const MEMBERS: GroupMember[] = [
  { connectionId: 'local', name: 'lifeos-coordinator' },
  { connectionId: 'local', name: 'notion-expert' },
  { connectionId: 'local', name: 'finance-coordinator' },
  { connectionId: 'local', name: 'health-coordinator' }
]

function userSays(text: string): GroupMessage[] {
  return [{ at: 0, from: { kind: 'user', name: 'andre' }, text }]
}

function names(members: GroupMember[]): string[] {
  return members.map(member => member.name)
}

describe('resolveGroupResponders — routing by relevance', () => {
  it('still sends an @-mentioned member alone', () => {
    expect(names(resolveGroupResponders(userSays('@notion-expert check this'), MEMBERS))).toEqual(['notion-expert'])
  })

  it('still wakes everyone for @everyone', () => {
    expect(resolveGroupResponders(userSays('@everyone standup'), MEMBERS)).toHaveLength(4)
  })

  it('routes an unmentioned domain question to the matching member', () => {
    // Every specialist that answers a question outside its domain pays a full
    // model call to say "(pass)". The room can spend that turn before the API
    // does, when the question names a domain unambiguously.
    const responders = resolveGroupResponders(
      userSays('as paginas do database Months tem o bloco de linked view Holdings?'),
      MEMBERS
    )

    expect(names(responders)).toContain('notion-expert')
    expect(names(responders)).not.toContain('health-coordinator')
    expect(names(responders)).not.toContain('finance-coordinator')
  })

  it('keeps the coordinator in the round so routing survives a wrong guess', () => {
    // Relevance is a heuristic over words. The coordinator is what makes a miss
    // recoverable: it can hand the task to whoever actually owns it, so a bad
    // guess costs a round rather than an unanswered question.
    const responders = resolveGroupResponders(userSays('quanto rendeu meu FGTS este mes?'), MEMBERS)

    expect(names(responders)).toContain('lifeos-coordinator')
    expect(names(responders)).toContain('finance-coordinator')
  })

  it('keeps deliberate domain stems without matching arbitrary substrings', () => {
    expect(names(resolveGroupResponders(userSays('preciso revisar o relatório financeiro'), MEMBERS))).toContain(
      'finance-coordinator'
    )
  })

  it('recognises a domain-neutral coordinator without requiring one profile name', () => {
    const members: GroupMember[] = [
      { connectionId: 'local', name: 'room-coordinator' },
      { connectionId: 'local', name: 'finance-coordinator' },
      { connectionId: 'local', name: 'health-coordinator' }
    ]

    expect(names(resolveGroupResponders(userSays('quanto rendeu meu FGTS?'), members))).toEqual([
      'room-coordinator',
      'finance-coordinator'
    ])
  })

  it('does not route on a domain term embedded inside another word', () => {
    const members: GroupMember[] = [
      { connectionId: 'local', name: 'notion-expert' },
      { connectionId: 'local', name: 'travel-planner' }
    ]

    expect(names(resolveGroupResponders(userSays('travel planner, please review my itinerary'), members))).toEqual([
      'notion-expert',
      'travel-planner'
    ])
  })

  it('does not route Notion from one ambiguous word', () => {
    expect(resolveGroupResponders(userSays('the mountain view was beautiful'), MEMBERS)).toHaveLength(4)
  })

  it('falls back to everyone when no domain is recognisable', () => {
    // Silence is the worst outcome. A question that matches nothing must still
    // reach the room rather than being routed into the void.
    expect(resolveGroupResponders(userSays('bom dia, tudo certo?'), MEMBERS)).toHaveLength(4)
  })

  it('routes on the whole exchange since the last user message', () => {
    const log: GroupMessage[] = [
      { at: 0, from: { kind: 'user', name: 'andre' }, text: 'preciso revisar os treinos da semana' },
      { at: 1, from: { kind: 'member', name: 'lifeos-coordinator' }, text: 'passando adiante' }
    ]

    expect(names(resolveGroupResponders(log, MEMBERS))).toContain('health-coordinator')
  })
})

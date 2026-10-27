/**
 * The roster's one-status-per-row derivation and the Needs you band built on
 * it. Pure over signals the pane already holds, so these cases drive the real
 * functions with plain snapshots: no gateway, no React.
 */

import { describe, expect, it } from 'vitest'

import { deriveRosterRows, type RosterStatusInput, rosterStatusResolvers } from './roster-pane-derivation'
import { type BotStatusSignals, deriveBotStatus, deriveGroupStatus } from './roster-status'
import type { BotMeta, RosterRow } from './types'

const bot = (name: string, extra: Partial<RosterRow> = {}): RosterRow =>
  ({
    name,
    connectionId: 'local',
    canonical_session: { id: `chat-${name}`, resolved_id: `chat-${name}` },
    ...extra
  }) as RosterRow

const signals = (over: Partial<RosterStatusInput['signals']> = {}): RosterStatusInput['signals'] => ({
  dots: {},
  prompts: {},
  routines: [],
  ...over
})

const input = (over: Partial<RosterStatusInput> = {}): RosterStatusInput => ({
  signals: signals(),
  attention: {},
  workingKeys: new Set(),
  activeConnectionId: 'local',
  allMeta: {},
  revealHidden: false,
  groupClarify: {},
  groupNeedsYou: {},
  groupRooms: {},
  ...over
})

function rows(roster: RosterRow[], status: RosterStatusInput, allMeta: Record<string, BotMeta> = {}) {
  const { statusOfBot, statusOfGroup } = rosterStatusResolvers({ ...status, allMeta })

  return deriveRosterRows({
    roster,
    allMeta,
    gatewayFilter: 'all',
    query: '',
    activityFilter: 'all',
    rowKindFilter: 'all',
    groupRooms: status.groupRooms,
    activeRosterKeys: new Set(),
    gatewayOptions: [],
    activityOf: () => 0,
    isPinned: () => false,
    statusOfBot,
    statusOfGroup
  })
}

const sectionNames = (out: ReturnType<typeof rows>) =>
  out.rosterRows.map(row => (row.kind === 'bot' ? row.bot.name : `#${row.name}`))

const bandNames = (out: ReturnType<typeof rows>) =>
  out.needsYouRows.map(row => (row.kind === 'bot' ? row.bot.name : `#${row.name}`))

describe('deriveBotStatus precedence', () => {
  it('shows exactly one state, loudest first: needs you > routine > working > failed > unread > idle', () => {
    const everything: BotStatusSignals = {
      dot: 'unread',
      prompt: { kind: 'clarify' as const, text: 'Which branch?' },
      failure: { reason: 'provider_quota_limit', message: 'out of credits' },
      routineRunning: true,
      working: true
    }

    expect(deriveBotStatus(everything)).toMatchObject({ kind: 'needs-you', reason: 'clarify', quote: 'Which branch?' })
    expect(deriveBotStatus({ ...everything, prompt: null }).kind).toBe('routine')
    expect(deriveBotStatus({ ...everything, prompt: null, routineRunning: false }).kind).toBe('working')
    expect(deriveBotStatus({ ...everything, prompt: null, routineRunning: false, working: false }).kind).toBe('failed')
    expect(deriveBotStatus({ dot: 'unread' }).kind).toBe('unread')
    expect(deriveBotStatus({}).kind).toBe('idle')
  })

  it('treats a refused delivery as a decision for the user, other failures as a quiet failed glyph', () => {
    expect(deriveBotStatus({ failure: { reason: 'agent_blocked', message: 'refused' } })).toMatchObject({
      kind: 'needs-you',
      reason: 'blocked',
      lifts: true
    })
    expect(deriveBotStatus({ failure: { reason: 'provider_auth_or_access', message: '401' } })).toMatchObject({
      kind: 'failed',
      lifts: false
    })
  })

  it('never lifts a row retained from an earlier paint, and lifts a hidden row only while revealed', () => {
    const waiting = { prompt: { kind: 'approval' as const, text: 'rm -rf build' } }

    expect(deriveBotStatus({ ...waiting, stale: true })).toMatchObject({ kind: 'needs-you', lifts: false, stale: true })
    expect(deriveBotStatus({ ...waiting, hidden: true }).lifts).toBe(false)
    expect(deriveBotStatus({ ...waiting, hidden: true, revealHidden: true }).lifts).toBe(true)
  })
})

describe('deriveGroupStatus', () => {
  it('keeps a member prompt distinct from an unread @user mention', () => {
    expect(deriveGroupStatus({ prompt: { kind: 'clarify', text: 'Ship it?' }, mentioned: true })).toMatchObject({
      reason: 'clarify',
      quote: 'Ship it?'
    })
    expect(deriveGroupStatus({ mentioned: true, mentionText: '@user look' })).toMatchObject({
      kind: 'needs-you',
      reason: 'mention'
    })
    expect(deriveGroupStatus({ running: true })).toMatchObject({ kind: 'working', lifts: false })
  })
})

describe('Needs you band', () => {
  const roster = [bot('alpha'), bot('beta'), bot('gamma')]

  it('is empty when nobody waits and every row stays in its section', () => {
    const out = rows(roster, input())

    expect(bandNames(out)).toEqual([])
    expect(sectionNames(out)).toEqual(['alpha', 'beta', 'gamma'])
  })

  it('lifts the waiting bot out of its section, quoting the question, and returns it once answered', () => {
    const waiting = input({
      signals: signals({
        dots: { 'chat-beta': 'needs-input' },
        prompts: { 'chat-beta': { kind: 'clarify', text: 'Deploy to prod?' } }
      })
    })

    const lifted = rows(roster, waiting)
    expect(bandNames(lifted)).toEqual(['beta'])
    expect(sectionNames(lifted)).toEqual(['alpha', 'gamma'])
    expect(rosterStatusResolvers(waiting).statusOfBot(roster[1])).toMatchObject({ quote: 'Deploy to prod?' })

    const cleared = rows(roster, input({ signals: signals({ dots: { 'chat-beta': 'working' } }) }))
    expect(bandNames(cleared)).toEqual([])
    expect(sectionNames(cleared)).toEqual(['alpha', 'beta', 'gamma'])
  })

  it('reads the prompt under the compression tip as well as the registry id', () => {
    const rotated = bot('beta', { canonical_session: { id: 'chat-root', resolved_id: 'chat-tip' } })
    const out = rows(
      [rotated],
      input({ signals: signals({ prompts: { 'chat-tip': { kind: 'approval', text: 'git push' } } }) })
    )

    expect(bandNames(out)).toEqual(['beta'])
  })

  it('greys but never lifts a remote row retained from an earlier paint', () => {
    const retained = bot('remote', { connectionId: 'box', remoteSource: true, sourceReachable: false })
    const status = input({ signals: signals({ dots: { 'chat-remote': 'needs-input' } }) })
    const out = rows([...roster, retained], status)

    expect(bandNames(out)).toEqual([])
    expect(rosterStatusResolvers(status).statusOfBot(retained)).toMatchObject({ kind: 'needs-you', stale: true })
  })

  it('lifts a hidden bot only while the hidden rows are revealed', () => {
    const allMeta = { gamma: { hidden: true } as BotMeta }
    const waiting = signals({ dots: { 'chat-gamma': 'needs-input' } })

    expect(bandNames(rows(roster, input({ signals: waiting }), allMeta))).toEqual([])

    const revealed = rows(roster, input({ signals: waiting, revealHidden: true }), allMeta)
    expect(bandNames(revealed)).toEqual(['gamma'])
    expect(revealed.filteredHiddenBots.map(row => row.name)).toEqual([])
  })

  it('marks a bot whose cron run the scheduler still owns as running a routine', () => {
    const status = input({ signals: signals({ routines: ['::alpha'] }) })

    expect(rosterStatusResolvers(status).statusOfBot(roster[0]).kind).toBe('routine')
    expect(rosterStatusResolvers(status).statusOfBot(roster[1]).kind).toBe('idle')
  })
})

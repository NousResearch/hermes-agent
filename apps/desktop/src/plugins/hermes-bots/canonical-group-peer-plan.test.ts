import { expect, it } from 'vitest'

import { groupCreationEngine, groupExecutionMode } from './canonical-group-capabilities'
import { PEER_HOME, PEER_MINI, peerRosterRow } from './canonical-group-contract-test-utils'

const here = { name: 'alpha', connectionId: 'local' }
const remote = peerRosterRow('mini', 'Mac Mini')
const targets = new Map([['mini', { ready: true as const, gateway: 'install:3f2a0000000000000000000000000000', installationId: 'install:3f2a0000000000000000000000000000', catalogDigest: PEER_MINI.room_link.catalog.catalog_digest, renewal: true }]])

it.each([
  ['owner need not lend', { ...PEER_HOME, room_link: { enabled: false } }, remote, targets, 'gateway-peer'],
  ['lender installation changed', PEER_HOME, { ...remote, installId: 'ffffffffffffffffffffffffffffffff' }, targets, 'unavailable'],
  ['owner ends with Desktop', { ...PEER_HOME, persistent_process: false }, remote, targets, 'desktop'],
  ['lender cannot invite', PEER_HOME, remote, new Map([['mini', { ready: false, reason: 'peer_invite_unsupported' }]]), 'desktop'],
  ['lender check failed', PEER_HOME, remote, new Map([['mini', { ready: false, reason: 'unreachable' }]]), 'unavailable'],
  ['lender check pending', PEER_HOME, remote, new Map(), 'checking'],
  ['nondefault remote Bot during a capability outage', PEER_HOME, peerRosterRow('mini', 'Mac Mini', 'coder'), new Map([['mini', { ready: false, reason: 'unreachable' }]]), 'desktop'],
  ['nondefault remote Bot', PEER_HOME, peerRosterRow('mini', 'Mac Mini', 'coder'), targets, 'desktop'],
  ['route disagrees with row', PEER_HOME, { ...remote, connectionId: 'spoofed' }, targets, 'unavailable'],
  ['local Bot is no longer seatable', { ...PEER_HOME, seatable_profiles: [] }, remote, targets, 'unavailable']
])('%s returns the engine and reason', (_case, owner, bot, peers, expected) => {
  const plan = groupCreationEngine('canonical', [here, bot], 'local', owner, peers as typeof targets)
  expect(plan.engine).toBe(expected)

  if (expected !== 'gateway-peer') {expect(plan.reason).toBeTruthy()}
})

it.each([false, true])('waits for every selected lender regardless of selection order (%s)', reverse => {
  const bots = [remote, peerRosterRow('nas', 'NAS')]
  const states = new Map([['mini', { ready: false as const, reason: 'unreachable' }]])
  expect(groupCreationEngine('canonical', [here, ...reverse ? bots.reverse() : bots], 'local', PEER_HOME, states).engine).toBe('checking')
})

it('keeps remote-only creation available on a legacy gateway', () => {
  const capabilities = { driver: false, persistent_process: false }
  const selection = [remote, peerRosterRow('mini', 'Mac Mini', 'coder')]
  expect(groupCreationEngine(groupExecutionMode(capabilities), selection, 'local', capabilities).engine).toBe('desktop')
})

it.each([false, true])('keeps a definitively incompatible selection Desktop-driven during another outage (%s)', reverse => {
  const bots = [remote, peerRosterRow('nas', 'NAS')]

  const states = new Map([
    ['mini', { ready: false as const, reason: 'unreachable' }],
    ['nas', { ready: false as const, reason: 'peer_invite_unsupported' }]
  ])

  expect(groupCreationEngine('canonical', [here, ...reverse ? bots.reverse() : bots], 'local', PEER_HOME, states).engine).toBe('desktop')
})

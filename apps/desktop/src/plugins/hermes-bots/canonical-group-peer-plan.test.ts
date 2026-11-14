import { expect, it } from 'vitest'

import { groupCreationEngine, groupExecutionMode } from './canonical-group-capabilities'
import { PEER_HOME, PEER_MINI, peerRosterRow } from './canonical-group-contract-test-utils'
import { peerTargetKey } from './canonical-group-peers'

const here = { name: 'alpha', connectionId: 'local' }
const remote = peerRosterRow('mini', 'Mac Mini')

const ready = (profile: string) => ({
  ready: true as const,
  gateway: 'install:3f2a0000000000000000000000000000',
  installationId: 'install:3f2a0000000000000000000000000000',
  catalogDigest: PEER_MINI.room_link.catalog.catalog_digest,
  renewal: true,
  profile
})

const mini = (profile = 'default') => peerTargetKey({ connectionId: 'mini', targetProfile: profile })
const targets = new Map([
  [mini(), ready('default')],
  [mini('coder'), ready('coder')]
])

it.each([
  ['owner need not lend', { ...PEER_HOME, room_link: { enabled: false } }, remote, targets, 'gateway-peer'],
  [
    'lender installation changed',
    PEER_HOME,
    { ...remote, installId: 'ffffffffffffffffffffffffffffffff' },
    targets,
    'unavailable'
  ],
  ['owner ends with Desktop', { ...PEER_HOME, persistent_process: false }, remote, targets, 'unavailable'],
  [
    'lender cannot invite',
    PEER_HOME,
    remote,
    new Map([[mini(), { ready: false, reason: 'peer_invite_unsupported' }]]),
    'unavailable'
  ],
  [
    'lender check failed',
    PEER_HOME,
    remote,
    new Map([[mini(), { ready: false, reason: 'unreachable' }]]),
    'unavailable'
  ],
  ['lender check pending', PEER_HOME, remote, new Map(), 'checking'],
  [
    'named remote Bot during a capability outage',
    PEER_HOME,
    peerRosterRow('mini', 'Mac Mini', 'coder'),
    new Map([[mini('coder'), { ready: false, reason: 'unreachable' }]]),
    'unavailable'
  ],
  ['named remote Bot', PEER_HOME, peerRosterRow('mini', 'Mac Mini', 'coder'), targets, 'gateway-peer'],
  [
    'named remote Bot whose profile its gateway has not answered for',
    PEER_HOME,
    peerRosterRow('mini', 'Mac Mini', 'coder'),
    new Map([[mini(), ready('default')]]),
    'checking'
  ],
  ['route disagrees with row', PEER_HOME, { ...remote, connectionId: 'spoofed' }, targets, 'unavailable'],
  ['local Bot is no longer seatable', { ...PEER_HOME, seatable_profiles: [] }, remote, targets, 'unavailable']
])('%s returns the engine and reason', (_case, owner, bot, peers, expected) => {
  const plan = groupCreationEngine('canonical', [here, bot], 'local', owner, peers as typeof targets)
  expect(plan.engine).toBe(expected)

  if (expected !== 'gateway-peer') {
    expect(plan.reason).toBeTruthy()
  }
})

it.each([false, true])('waits for every selected lender regardless of selection order (%s)', reverse => {
  const bots = [remote, peerRosterRow('nas', 'NAS')]
  const states = new Map([[mini(), { ready: false as const, reason: 'unreachable' }]])
  expect(
    groupCreationEngine('canonical', [here, ...(reverse ? bots.reverse() : bots)], 'local', PEER_HOME, states).engine
  ).toBe('checking')
})

it('never plans a Desktop-driven group chat on a gateway too old to run group chats', () => {
  const capabilities = { driver: false, persistent_process: false }
  const selection = [remote, peerRosterRow('mini', 'Mac Mini', 'coder')]
  expect(groupCreationEngine(groupExecutionMode(capabilities), selection, 'local', capabilities)).toEqual({
    engine: 'unavailable',
    reason: 'owner_update_required'
  })
})

it.each([false, true])('names the first lender that is missing something once every check answered (%s)', reverse => {
  const bots = [remote, peerRosterRow('nas', 'NAS')]

  const states = new Map([
    [mini(), { ready: false as const, reason: 'unreachable' }],
    [
      peerTargetKey({ connectionId: 'nas', targetProfile: 'default' }),
      { ready: false as const, reason: 'peer_invite_unsupported' }
    ]
  ])

  const plan = groupCreationEngine(
    'canonical',
    [here, ...(reverse ? bots.reverse() : bots)],
    'local',
    PEER_HOME,
    states
  )
  expect(plan.engine).toBe('unavailable')
  expect(plan.connectionId).toBe(reverse ? 'nas' : 'mini')
})

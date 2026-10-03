/** Fake computers for the continuation tests: each answers for its own installation, refuses what it doesn't hold
 * the way a gateway does, and can go offline. Status payloads follow `groups.succession.status`. */
import { CANONICAL_GROUP_CAPABILITIES } from './group-test-utils'

export const hex = (character: string) => character.repeat(32)
export const MINI = `install:${hex('a')}`, VPS = `install:${hex('b')}`, LAPTOP = `install:${hex('c')}`, GUEST = `install:${hex('e')}`
export const binding = { connectionId: 'mac-mini', profile: 'default', roomId: 'room-harbor' }

export const registry = [
  { id: 'mac-mini', label: 'Mac mini', installId: hex('a') }, { id: 'vps', label: 'Home VPS', installId: hex('b') },
  { id: 'laptop', label: 'Laptop', installId: hex('c') }, { id: 'unrelated', label: 'Work box', installId: hex('d') }
]

export const LAYER7 = ['groups.succession.status', 'groups.succession.prepare', 'groups.succession.promote', 'groups.succession.keep',
  'groups.succession.branch_log', 'groups.succession.move', 'groups.succession.continue_anyway', 'groups.succession.learn',
  'groups.succession.move_now',
  'groups.custody.designate', 'groups.custody.allow', 'groups.custody.add', 'groups.custody.remove', 'groups.custody.automatic',
  'groups.custody.status']

export const members = [{ member_id: 'atlas', profile: 'default', handle: 'atlas', display_name: 'Atlas Bot' },
  { member_id: 'mira', profile: 'default', handle: 'mira', display_name: 'Mira Bot' }]

export type Handler = (method: string, params: Record<string, unknown>) => unknown

export const capabilities = (install: string, layer7 = true) => ({ ...CANONICAL_GROUP_CAPABILITIES, authority_gateway_id: install,
  methods: [...CANONICAL_GROUP_CAPABILITIES.methods, ...layer7 ? LAYER7 : []] })

export const roomState = (extra: Record<string, unknown> = {}, epoch = 1) => ({ room: { name: 'Harbor launch', authority_epoch: epoch, members },
  driver_status: { running: true, working: false, counts: {}, pending_actions: [], ...extra } })

export const backup = (install: string, name: string | null, extra: Record<string, unknown> = {}) => ({ install_id: install, name,
  successor: true, allowed: true, designated: true, kind: 'member', operator_name: 'Dana', readiness: 'caught_up', behind_by: 0, last_seen: null, ...extra })

export const status = (extra: Record<string, unknown> = {}) => ({
  state: 'ok', host: { install_id: MINI, name: 'Mac mini', reachable: true, since: null },
  this_install: { install_id: MINI, name: 'Mac mini', role: 'host' }, owner: { name: 'Dana' },
  backups: [backup(VPS, 'Home VPS'), backup(LAPTOP, 'Laptop', { readiness: 'behind', behind_by: 3 })],
  at_risk: { count: 0 }, moving: null, conflict: null, moved: null, work: null, actions: [], unavailable_reason: null,
  previous_host: null, unavailable_bots: [], last_attempt: null, ...extra
})

export const offlineStatus = (extra: Record<string, unknown> = {}) => status({
  state: 'host_unreachable', host: { install_id: MINI, name: 'Mac mini', reachable: false, since: 1_700_000_000 },
  this_install: { install_id: VPS, name: 'Home VPS', role: 'backup' },
  actions: [{ action: 'continue', targets: [VPS, LAPTOP] }], ...extra
})

export const refusal = (reason: string, data: Record<string, unknown> = {}) => Object.assign(new Error(reason), { code: 4001, data: { reason, ...data } })

/** One computer: its own installation, a room it hosts or keeps, and anything else refused like the gateway does. */
export function computer(install: string, methods: Record<string, Handler>, layer7 = true): Handler {
  return (method, params) => {
    if (Object.hasOwn(methods, method)) {return methods[method](method, params)}

    if (method === 'groups.capabilities') {return capabilities(install, layer7)}
    throw refusal('room_not_found')
  }
}

/** Capabilities that also say who runs the computer and whether it stays on (`room_identity`). */
export const runBy = (install: string, operator: string | null, alwaysOn = false): Handler => () =>
  ({ ...capabilities(install), room_identity: { operator_name: operator, always_on: alwaysOn } })

export const unreachable: Handler = () => {throw new Error('connection lost')}

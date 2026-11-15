/**
 * The gateway's declared `groups.*` contract: the OpenRPC document generated from
 * `tui_gateway/contracts` (`tests/tui_gateway/contracts/test_generated.py` fails when it is stale,
 * and `tests/gateway/test_canonical_wire_contract.py` when the gateway serves an undeclared group
 * method). A method or field the gateway renames or drops fails every test that captures the
 * client's requests.
 */

import { readFileSync } from 'node:fs'
import { join } from 'node:path'

import type * as HermesSdk from '@hermes/plugin-sdk'
// eslint-disable-next-line no-restricted-imports -- Wire fixtures use the real client decoder, not a plugin capability.
import { jsonRpcErrorFromFrame } from '@hermes/shared'
import { afterEach } from 'vitest'

import type { CanonicalRoom } from './canonical-groups'
import peerControlWire from './fixtures/peer-control-responses.json'

interface Schema {
  properties: Record<string, unknown>
  required?: string[]
}

interface Declaration {
  name: string
  params: [{ schema: { $ref: string } }]
  result: { schema: { $ref: string } }
}

const contract = JSON.parse(
  readFileSync(join(process.cwd(), '..', 'shared', 'src', 'gateway-contract.openrpc.json'), 'utf8')
)

const schema = (ref: string): Schema => contract.components.schemas[ref.split('/').at(-1)!]

// The gateway checks a method against its own (`x-canonical-methods`) declaration first, then the
// shared one (gateway/session_controls.py), so the canonical entries are read last and win.
const declarations = new Map<string, Declaration>(
  [...contract.methods, ...contract['x-canonical-methods']].map((method: Declaration) => [method.name, method])
)

export interface DeclaredFields {
  fields: readonly string[]
  required: readonly string[]
}

const fields = ({ properties, required = [] }: Schema): DeclaredFields => ({ fields: Object.keys(properties), required })

/** `method`'s declared params and result keys, or undefined when the contract does not declare it. */
export function declaredGroupMethod(method: string): { params: DeclaredFields; result: DeclaredFields } | undefined {
  const declaration = declarations.get(method)

  return declaration && {
    params: fields(schema(declaration.params[0].schema.$ref)),
    result: fields(schema(declaration.result.schema.$ref))
  }
}

/** A `groups.capabilities` answer fixture, refused when it lists an undeclared method or a key
 *  the declared answer does not have. */
export function declaredCapabilities<T extends { methods: string[] }>(answer: T): T {
  const declared = declaredGroupMethod('groups.capabilities')!.result

  const problems = [
    ...answer.methods.filter(method => !declaredGroupMethod(method)),
    ...Object.keys(answer).filter(key => !declared.fields.includes(key)),
    ...declared.required.filter(key => !Object.hasOwn(answer, key))
  ]

  if (problems.length) {
    throw new Error(`groups.capabilities fixture does not fit the gateway contract: ${problems.join(', ')}`)
  }

  return answer
}

/** `groups.capabilities` as `main`'s gateway answers it with a running driver
 *  (gateway/session_group_controls.py: `methods` is its `GROUP_METHODS`). */
export const CANONICAL_GROUP_CAPABILITIES = declaredCapabilities({
  protocol_version: 2,
  driver: true,
  persistent_process: true,
  authority_gateway_id: 'install:canonical',
  room_link: { enabled: false, reason: 'canonical_driver_required' },
  features: ['room_identity', 'monotonic_log', 'replayable_disband'],
  methods: ['groups.capabilities', 'groups.list', 'groups.state', 'groups.log', 'groups.create', 'groups.rename', 'groups.context',
    'groups.disband', 'groups.send', 'groups.attachment.upload', 'groups.attachment.download', 'groups.stop',
    'groups.retry', 'groups.discard', 'groups.approve'],
  max_log_limit: 500
})

const CATALOG_DIGEST = 'c'.repeat(64)

/** `groups.capabilities` of a gateway that also hosts Bots of other gateways' rooms and seats theirs
 *  (gateway/session_group_peers.py): the `groups.peer.*` methods, `peer_renewal_horizon`, and a
 *  `room_link` that is enabled, with the catalog a room pins its member to. */
export function peerGroupCapabilities(gateway = 'install:peer') {
  const endpoint = { available: true, url: `https://${gateway.slice(8)}.example.ts.net:8642`, transport_security: 'tls' }

  return declaredCapabilities({
    ...CANONICAL_GROUP_CAPABILITIES,
    authority_gateway_id: gateway,
    room_link: {
      enabled: true, profile: 'default', endpoint,
      catalog: { installation_id: gateway, protocol_versions: [2], link_modes: ['direct'], persistent_process: true, text: true,
        attachments: false, execution_policy: { target_profile: 'default', policy_digest: 'p'.repeat(64) },
        catalog_digest: CATALOG_DIGEST, endpoint }
    },
    features: [...CANONICAL_GROUP_CAPABILITIES.features, 'peer_renewal_horizon'],
    methods: [...CANONICAL_GROUP_CAPABILITIES.methods, 'groups.peer.register', 'groups.peer.invite', 'groups.peer.revoke']
  })
}

/** The doubles of a home (`install:a18b0000000000000000000000000000`) seating the default Bot of `install:3f2a0000000000000000000000000000`. */
export const PEER_HOME = peerGroupCapabilities('install:a18b0000000000000000000000000000')
export const PEER_MINI = peerGroupCapabilities('install:3f2a0000000000000000000000000000')
export const PEER_REGISTERED = { registered: true, mode: 'direct', transport_security: 'tls', target_install_id: 'install:3f2a0000000000000000000000000000', target_profile: 'default' }
export const peerInvitation = (grant = 'grant', endpoint: unknown = PEER_MINI.room_link.endpoint) =>
  ({ grant, target_profile: 'default', catalog: PEER_MINI.room_link.catalog, endpoint })

const peerControlError = (key: keyof typeof peerControlWire) => jsonRpcErrorFromFrame(JSON.parse(peerControlWire[key]).error)

/** Decode the wire errors as Desktop does; an untyped RuntimeError loses its text in ws.py. */
export const DISBAND_FAILURES = [
  { name: 'internal error', error: peerControlError('internalError'), detail: 'disbandUnconfirmed' },
  { name: 'peer unreachable', error: peerControlError('peer_unreachable'), detail: 'disbandPeerUnreachable' },
  { name: 'stop pending', error: peerControlError('room_stop_pending'), detail: 'disbandStopPending' },
  { name: 'already disbanding', error: peerControlError('disbanding'), detail: 'disbandInProgress' },
  { name: 'held grants', error: peerControlError('room_disbanding_retry'), detail: 'disbandInProgress' },
  { name: 'revoke unsupported', error: peerControlError('peer_revoke_unsupported'), detail: 'disbandRevokeUnsupported' },
  { name: 'service required', error: peerControlError('runtime_coordination_required'), detail: 'disbandServiceRequired' },
  { name: 'legacy stopping', error: jsonRpcErrorFromFrame({ code: 5114, message: 'room work is still stopping; retry deletion after Stop completes' }), detail: 'disbandStopPending' },
  { name: 'short stopping', error: jsonRpcErrorFromFrame({ code: 5114, message: 'room work is still stopping' }), detail: 'disbandStopPending' },
  { name: 'other refusal', error: jsonRpcErrorFromFrame({ code: 4001, message: 'Disband refused', data: { reason: 'invalid_params' } }), detail: 'disbandUnconfirmed' },
  { name: 'connection closed', error: new Error('Connection closed'), detail: 'disbandUnconfirmed' }
] as const

/** A roster row for Bot `name` on another connection. */
export const peerRosterRow = (connectionId: string, connectionLabel: string, name = 'default') => ({
  name, connectionId, connectionLabel, remoteSource: true, sourceScoped: true,
  installId: connectionId === 'mini' ? '3f2a0000000000000000000000000000' : 'b27c0000000000000000000000000000',
  route: { connectionId, mode: 'remote' as const, profile: name, targetProfile: name }
})

type Answer = (method: string, params: Record<string, any>) => unknown

/** Answer each `request(route, method, params)` from `answers[route.connectionId]` (`local` without a
 *  route): a value, a promise, or an Error to reject with. Any other request fails the test. */
export function answerGroupRequests(request: { mockImplementation: (fn: ProfileRequest) => unknown }, answers: Record<string, Answer>) {
  request.mockImplementation(async (route, method, params) => {
    const connection = (route as { connectionId?: string } | null)?.connectionId ?? 'local'
    const answer = await answers[connection]?.(method, params ?? {})

    if (answer === undefined) {throw new Error(`Unexpected RPC: ${connection} ${method}`)}

    if (answer instanceof Error) {throw answer}

    return answer
  })
}

/** The standalone server's answer (`hermes serve`, `hermes dashboard`; tui_gateway/methods_groups.py
 *  `_METHODS`): it starts its own room driver at boot, so `driver` is true too. */
export const STANDALONE_GROUP_CAPABILITIES = declaredCapabilities({
  protocol_version: 2,
  driver: true,
  persistent_process: false,
  authority_gateway_id: 'install:standalone',
  room_link: { enabled: false, reason: 'gateway_roomlink_secret_unavailable' },
  features: ['authority_epoch', 'coordinator_fencing', 'room_identity', 'monotonic_log', 'idempotent_send',
    'replayable_disband', 'typed_events', 'actor_identity', 'log_replication', 'authority_takeover'],
  methods: ['groups.capabilities', 'groups.list', 'groups.create', 'groups.state', 'groups.send', 'groups.rename', 'groups.context',
    'groups.log', 'groups.disband', 'groups.replicate', 'groups.replica_state', 'groups.promote', 'groups.demote',
    'groups.stop', 'groups.retry', 'groups.approve', 'groups.peer.invite', 'groups.peer.revoke', 'groups.peer.register',
    'groups.attachment.upload', 'groups.attachment.download'],
  max_log_limit: 500
})

/** Older standalone backends still connect: they advertise neither file method. */
export const LEGACY_STANDALONE_GROUP_CAPABILITIES = declaredCapabilities({
  ...STANDALONE_GROUP_CAPABILITIES,
  methods: STANDALONE_GROUP_CAPABILITIES.methods.filter(method => !method.startsWith('groups.attachment.'))
})

export interface GroupRequestCall {
  method: string
  params: Record<string, unknown>
}

const shared = new Map<string, Declaration>(contract.methods.map((method: Declaration) => [method.name, method]))

/** The keys a call to the standalone server would be refused for (`4000`, tui_gateway/rpc_dispatch.py
 *  `validate_params`): it checks only for unknown keys, against the shared declaration. Methods it
 *  does not list are never offered there. */
export function standaloneGroupProblems({ method, params }: GroupRequestCall): string[] {
  const declaration = STANDALONE_GROUP_CAPABILITIES.methods.includes(method) ? shared.get(method) : undefined
  const { properties } = declaration ? schema(declaration.params[0].schema.$ref) : { properties: undefined }

  return properties ? Object.keys(params).filter(key => params[key] !== undefined && !Object.hasOwn(properties, key))
    .map(key => `${method}.${key}`) : []
}

/** Refuse what either surface refuses before any handler runs. The gateway (`4001 invalid_params`):
 *  a method it does not list, an unknown key, or a missing required one. The standalone server: an
 *  unknown key in a method it lists. */
export function assertCanonicalGroupCalls(calls: GroupRequestCall[]): void {
  for (const { method, params } of calls) {
    const declared = peerGroupCapabilities().methods.includes(method) ? declaredGroupMethod(method)?.params : undefined

    if (!declared) {
      throw new Error(`Undeclared canonical group method: ${method}`)
    }

    // A key whose value is undefined never reaches the wire.
    const sent = Object.keys(params).filter(field => params[field] !== undefined)

    for (const field of sent) {
      if (!declared.fields.includes(field)) {
        throw new Error(`Undeclared canonical group field: ${method}.${field}`)
      }
    }

    for (const field of declared.required) {
      if (!sent.includes(field)) {
        throw new Error(`Missing canonical group field: ${method}.${field}`)
      }
    }

    // The renewal horizon is sent only when the lender advertises peer_renewal_horizon.
    const refused = standaloneGroupProblems({ method, params }).filter(field => field !== 'groups.peer.invite.status_ttl_seconds')

    if (refused.length) {
      throw new Error(`The standalone server refuses group field: ${refused.join(', ')}`)
    }
  }
}

type ProfileRequest = (route: unknown, method: string, params?: Record<string, unknown>, ...rest: unknown[]) => unknown

/** Wrap a test's `host.requestProfile` double: its answers are unchanged, and
 *  every `groups.*` call it saw is checked against both surfaces after each test. */
export function captureGroupRequests(request: ProfileRequest) {
  const calls: GroupRequestCall[] = []
  afterEach(() => assertCanonicalGroupCalls(calls.splice(0)))

  return {
    calls,
    request: (...args: Parameters<ProfileRequest>) => {
      const [, method, params] = args

      if (method.startsWith('groups.')) {
        calls.push({ method, params: { ...params } })
      }

      return request(...args)
    }
  }
}

/** `@hermes/plugin-sdk` for a dialog or gate on the `local` connection's default profile, with its
 *  group requests going to `request` (contract-checked), Bot Mode words from `translate()` (read at
 *  render, since the plugin's bundle imports this module) and `host` overrides. */
export async function localGatewaySdkMock(sdk: typeof HermesSdk, request: ProfileRequest, translate: () => unknown,
  host: Record<string, unknown> = {}) {
  const { en } = await import('@/i18n/en')

  return {
    ...sdk,
    gatewayActivationEpoch: () => 1,
    useGatewayActivationEpoch: () => 1,
    host: {
      ...sdk.host, requestProfile: captureGroupRequests(request).request,
      request: (method: string, params?: Record<string, unknown>) => request(null, method, params),
      state: { ...sdk.host.state, connectionId: sdk.atom<string | null>('local'), profile: sdk.atom('default'), gateway: sdk.atom('open') },
      ...host
    },
    useI18n: () => ({ locale: 'en', t: en }),
    usePluginI18n: () => translate()
  }
}

/** Mirrors hosted_rooms.create_room: an ID cannot change payload or resurrect an ended room. */
export function strictGroupCreation() {
  const rooms = new Map<string, CanonicalRoom>()

  const create = (params: Record<string, any>) => {
    const existing = rooms.get(params.room_id)

    if (existing?.disbanded_at != null) {throw new Error('room_id belongs to a disbanded room')}

    if (existing && (existing.name !== params.name || JSON.stringify(existing.members) !== JSON.stringify(params.members))) {
      throw new Error('room_id already exists with different state')
    }

    const room = existing ?? { room_id: params.room_id, name: params.name, members: structuredClone(params.members),
      authority_gateway_id: PEER_HOME.authority_gateway_id, authority_epoch: 1 }

    rooms.set(room.room_id, room)

    return { room }
  }

  return { rooms, create }
}

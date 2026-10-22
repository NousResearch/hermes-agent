/**
 * The gateway's declared `groups.*` contract: the OpenRPC document generated from
 * `tui_gateway/contracts` (`tests/tui_gateway/contracts/test_generated.py` fails when it is stale,
 * and `tests/gateway/test_canonical_wire_contract.py` when the gateway serves an undeclared group
 * method). A method or field the gateway renames or drops fails every test that captures the
 * client's requests.
 */

import { readFileSync } from 'node:fs'
import { join } from 'node:path'

import { afterEach } from 'vitest'

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
  methods: ['groups.capabilities', 'groups.list', 'groups.state', 'groups.log', 'groups.create', 'groups.rename',
    'groups.disband', 'groups.send', 'groups.attachment.upload', 'groups.attachment.download', 'groups.stop',
    'groups.retry', 'groups.discard', 'groups.approve'],
  max_log_limit: 500
})

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
  methods: ['groups.capabilities', 'groups.list', 'groups.create', 'groups.state', 'groups.send', 'groups.rename',
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
    const declared = CANONICAL_GROUP_CAPABILITIES.methods.includes(method) ? declaredGroupMethod(method)?.params : undefined

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

    const refused = standaloneGroupProblems({ method, params })

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

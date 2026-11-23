import { gatewayActivationEpoch, host, useGatewayActivationEpoch, useValue } from '@hermes/plugin-sdk'
import { useEffect, useState } from 'react'

import { peerRoute, peerTargetKey, type PeerTargetState, readPeerTargets } from './canonical-group-peers'
import { canonicalGroupRequest } from './canonical-groups'
import type { GroupEngine } from './group-engine'
import { resolveBotConnectionRoute } from './routing'
import type { RosterRow } from './types'

export type GroupExecutionMode = 'canonical' | 'legacy' | 'unavailable'

const stringList = (value: unknown): value is string[] | undefined =>
  value === undefined || (Array.isArray(value) && value.every(item => typeof item === 'string'))

export function groupExecutionMode(value: unknown): GroupExecutionMode {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return 'unavailable'
  }

  const capability = value as Record<string, unknown>
  const { driver, persistent_process, features, methods } = capability

  if (!stringList(features) || !stringList(methods)) {
    return 'unavailable'
  }

  if (driver === true) {
    return 'canonical'
  }

  // App-managed hosted owners can be nonpersistent, even when their driver stops.
  const ownsRooms =
    Object.hasOwn(capability, 'protocol_version') ||
    Object.hasOwn(capability, 'authority_gateway_id') ||
    methods?.includes('groups.create') ||
    features?.includes('canonical_session_owner')

  if (driver === false && persistent_process === false && !ownsRooms) {
    return 'legacy'
  }

  return 'unavailable'
}

/** Two surfaces answer `groups.capabilities`: the gateway's own (gateway/session_group_controls.py)
 *  and the standalone server (`hermes serve`, `hermes dashboard`; tui_gateway/methods_groups.py).
 *  Standalone rooms support files; Discard is advertised only when forwarded to an owning gateway.
 *  A room offers a control only when its gateway lists the control's method: `undefined` until a
 *  read succeeds (none), `null` when a successful answer lists no methods (every control, as before). */
export type GroupMethods = ReadonlySet<string> | null | undefined

export function groupAdvertisedMethods(value: unknown): ReadonlySet<string> | null {
  const methods = value && typeof value === 'object' ? (value as Record<string, unknown>).methods : undefined

  return Array.isArray(methods) && stringList(methods) ? new Set(methods) : null
}

export const offersGroupMethod = (methods: GroupMethods, method: string) =>
  methods === null || Boolean(methods?.has(method))

/** Stop and Deny only end work, and an older gateway answers an unknown method harmlessly, so they
 *  stay offered while the method list is still unknown (`undefined`); a read that lists methods
 *  without them still hides them. */
export const offersGroupSafetyMethod = (methods: GroupMethods, method: string) =>
  methods === undefined || offersGroupMethod(methods, method)

// `host.state.gateway` is the foreground connection's socket: it says nothing about a room on
// another connection. A host double without socket state reads as one socket that stays open.
const onForeground = (connectionId: string) => (host.state?.connectionId?.get() ?? connectionId) === connectionId

/** The group methods `route`'s gateway lists, read once per opening of the gateway socket: an
 *  update restarts the gateway, so a reopened socket may be a gateway with other methods. Until a
 *  read for the current opening succeeds it lists none, so a failed read never offers a control the
 *  gateway did not confirm, and a reply from before the socket changed never publishes. While the
 *  room is visible and the socket open, a failed read is tried again every two seconds. */
export function useGroupMethods(route: { connectionId: string; profile: string }, visible = true): GroupMethods {
  const [opening, setOpening] = useState(0)
  const [answer, setAnswer] = useState<{ opening: number; methods: ReadonlySet<string> | null } | null>(null)
  const [attempt, setAttempt] = useState(0)
  const answered = answer?.opening === opening

  const { connectionId } = route

  useEffect(
    () =>
      host.state?.gateway?.listen(() => {
        if (onForeground(connectionId)) {
          setOpening(value => value + 1)
        }
      }),
    [connectionId]
  )

  useEffect(() => {
    if (answered || !visible || (onForeground(connectionId) && (host.state?.gateway?.get() ?? 'open') !== 'open')) {
      return
    }
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | undefined
    void canonicalGroupRequest<unknown>(route, 'groups.capabilities').then(
      value => {
        if (!cancelled) {
          setAnswer({ opening, methods: groupAdvertisedMethods(value) })
        }
      },
      () => {
        if (!cancelled) {
          timer = setTimeout(() => setAttempt(value => value + 1), 2000)
        }
      }
    )

    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [route, connectionId, visible, answered, attempt, opening])

  return answered ? answer.methods : undefined
}

/** The profiles `groups.create` will seat for THIS caller, as `groups.capabilities`
 *  reports them (the server's member rules: parked, retiring and unaddressable bots
 *  are out, and only a host-scoped socket seats served siblings). Null when the
 *  gateway predates the field, so callers keep their connection-only filter. */
export function groupSeatableProfiles(value: unknown): null | Set<string> {
  const seatable = value && typeof value === 'object' ? (value as Record<string, unknown>).seatable_profiles : undefined

  if (!Array.isArray(seatable)) {
    return null
  }

  return new Set(seatable.filter((name): name is string => typeof name === 'string'))
}

/** Fence a group-creation flow to the source that approved it. Pass the epoch the
 *  capability was read under (default: now); the predicate turns false once the
 *  connection, profile, socket or activation epoch moves, so a late result is
 *  neither acted on nor published. */
export function groupCreationSource(
  route: { connectionId: string; profile: string },
  activationEpoch = gatewayActivationEpoch()
) {
  return () =>
    gatewayActivationEpoch() === activationEpoch &&
    route.connectionId === host.state.connectionId.get() &&
    route.profile === host.state.profile.get() &&
    host.state.gateway.get() === 'open'
}

export interface GroupCapabilityState {
  activationEpoch: number
  snapshot: unknown
  mode: GroupExecutionMode
  seatable: null | Set<string>
  source: string
}

/** `groups.capabilities` for the active source while `active`, keyed by that
 *  source so an answer from a source the window has left is never used. */
export function useGroupCapability(active: boolean, attempt = 0) {
  const connectionId = useValue(host.state.connectionId)
  const profile = useValue(host.state.profile)
  const gateway = useValue(host.state.gateway)
  const activationEpoch = useGatewayActivationEpoch()
  const source = JSON.stringify([connectionId, profile, gateway, activationEpoch])
  const [capability, setCapability] = useState<GroupCapabilityState | null>(null)

  useEffect(() => {
    let cancelled = false
    setCapability(null)

    if (!active || gateway !== 'open' || !connectionId) {
      return
    }

    const route = { connectionId, profile }
    const sourceCurrent = groupCreationSource(route, activationEpoch)

    const settle = (result: unknown, failed: boolean) => {
      if (!cancelled && sourceCurrent()) {
        setCapability({
          activationEpoch,
          snapshot: result,
          mode: failed ? 'unavailable' : groupExecutionMode(result),
          seatable: failed ? null : groupSeatableProfiles(result),
          source
        })
      }
    }

    void canonicalGroupRequest<unknown>(route, 'groups.capabilities').then(
      result => settle(result, false),
      () => settle(null, true)
    )

    return () => {
      cancelled = true
    }
  }, [activationEpoch, active, attempt, connectionId, gateway, profile, source])

  const current = capability?.source === source ? capability : null

  return {
    capability,
    connectionId,
    mode:
      !connectionId || gateway !== 'open' ? ('unavailable' as const) : current ? current.mode : ('checking' as const),
    observedRoute: connectionId ? { connectionId, profile } : null,
    seatable: current ? current.seatable : null,
    source
  }
}

/** True when `bot` lives on `connectionId`: its row and its route (when it
 *  carries one) both name that connection. A row whose route disagrees with its
 *  connection is on neither side. */
export function botOnConnection(bot: RosterRow, connectionId: null | string): boolean {
  const connection = String(connectionId || '').trim()

  if (!connection) {
    return false
  }

  const routeAgrees = !bot.route?.connectionId || bot.route.connectionId === connection

  return bot.connectionId ? bot.connectionId === connection && routeAgrees : !bot.remoteSource && routeAgrees
}

/** A bot on another connection whose own gateway this Desktop reaches: a live row whose route
 *  resolves to the same connection the row names. */
function peerReachable(bot: RosterRow): boolean {
  if (bot.sourceMissing || bot.sourceReachable === false) {
    return false
  }

  const { route } = resolveBotConnectionRoute(bot)

  return Boolean(route && (!bot.connectionId || route.connectionId === bot.connectionId))
}

function memberConnection(bot: RosterRow, connectionId: null | string): string {
  return String(bot.connectionId || bot.route?.connectionId || connectionId || '').trim()
}

const spansConnections = (members: RosterRow[], connectionId: null | string) =>
  new Set(members.map(bot => memberConnection(bot, connectionId))).size > 1

/** A name this Desktop has a room record for: one that seated members, kept messages or holds a
 *  room id. Bots' own memberships write none, a disband leaves only a tombstone, and a layout write
 *  (section, order, pin) on a membership-only name leaves an empty one, so such a name is still a
 *  new group chat. */
export function groupRoomRecorded(
  room: { tombstone?: boolean; members?: unknown[]; log?: unknown[]; roomId?: null | string } | undefined
): boolean {
  return Boolean(room && !room.tombstone && (room.members?.length || room.log?.length || room.roomId))
}

/** True when this Desktop drives the room recorded as `room`. No new group chat is ever made this
 *  way: only one saved as `engine: 'desktop'` by an older build, or an unmarked older record whose
 *  members span connections, keeps its history and composer in the Desktop-driven view. A name
 *  with no record is a new group chat, which a gateway runs. */
export function groupDrivenByDesktop(
  room:
    | { engine?: GroupEngine; tombstone?: boolean; members?: unknown[]; log?: unknown[]; roomId?: null | string }
    | undefined,
  members: RosterRow[],
  connectionId: null | string
): boolean {
  return room?.engine ? room.engine === 'desktop' : groupRoomRecorded(room) && spansConnections(members, connectionId)
}

/** True when every bot in `selection` lives on ONE connection other than `connectionId`: that
 *  connection's gateway, not this one, should own the group chat, so creation from here is refused
 *  and the person adds a bot from this connection or switches to that one. */
export function groupSelectionElsewhere(selection: RosterRow[], connectionId: null | string): boolean {
  return (
    selection.length > 0 && !spansConnections(selection, connectionId) && !botOnConnection(selection[0], connectionId)
  )
}

/** Every new group chat is run by a gateway (`gateway`: bots on this connection only; `gateway-peer`:
 *  bots on other computers too, seated through their own gateways). `unavailable` names what is
 *  missing (`reason`) and on which connection, so the dialog can say so and offer Retry. */
export type GroupCreationEngine = 'elsewhere' | 'gateway' | 'gateway-peer' | 'checking' | 'unavailable'
export interface GroupCreationPlan {
  engine: GroupCreationEngine
  reason?: string
  connection?: string
  connectionId?: string
}

/** Owning peers does not require the owner to lend its own Bot through RoomLink. */
export function groupSeatsPeers(value: unknown): boolean {
  const answer = value as { driver?: boolean; persistent_process?: boolean; methods?: string[] } | null

  return (
    answer?.driver === true &&
    answer.persistent_process === true &&
    Array.isArray(answer.methods) &&
    answer.methods.includes('groups.peer.register')
  )
}

/** The one decision point for creating a group chat: always on a gateway, never Desktop-driven. */
export function groupCreationEngine(
  mode: GroupExecutionMode | 'checking',
  selection: RosterRow[],
  connectionId: null | string,
  snapshot?: unknown,
  targets: ReadonlyMap<string, PeerTargetState> = new Map()
): GroupCreationPlan {
  if (mode === 'checking') {
    return { engine: 'checking', reason: 'capabilities_pending' }
  }

  if (!connectionId || mode === 'unavailable') {
    return { engine: 'unavailable', reason: 'owner_unavailable' }
  }

  // A gateway too old to run group chats cannot own one: the person updates Hermes there.
  if (mode === 'legacy') {
    return { engine: 'unavailable', reason: 'owner_update_required' }
  }

  if (selection.some(bot => !botOnConnection(bot, connectionId) && !peerReachable(bot))) {
    return { engine: 'unavailable', reason: 'route_unavailable' }
  }

  if (groupSelectionElsewhere(selection, connectionId)) {
    return { engine: 'elsewhere', reason: 'other_connection' }
  }
  const seatable = groupSeatableProfiles(snapshot)

  if (
    selection.some(
      bot =>
        botOnConnection(bot, connectionId) &&
        seatable &&
        !seatable.has(bot.route?.targetProfile ?? bot.targetProfile ?? bot.name)
    )
  ) {
    return { engine: 'unavailable', reason: 'profile_unseatable' }
  }

  if (!spansConnections(selection, connectionId)) {
    return { engine: 'gateway' }
  }

  if (!groupSeatsPeers(snapshot)) {
    return { engine: 'unavailable', reason: 'peer_owner_unavailable' }
  }

  const plans = selection
    .filter(member => !botOnConnection(member, connectionId))
    .map(bot => peerCreationPlan(bot, connectionId, targets))

  // Finish every lender read first, then name the first one that is missing something.
  return (
    plans.find(plan => plan.engine === 'checking') ??
    plans.find(plan => plan.engine === 'unavailable') ?? { engine: 'gateway-peer' }
  )
}

function peerCreationPlan(
  bot: RosterRow,
  connectionId: string,
  targets: ReadonlyMap<string, PeerTargetState>
): GroupCreationPlan {
  const route = peerRoute(bot, connectionId)
  const connection = connectionName(bot)
  const location = { connection, connectionId: route?.connectionId ?? bot.connectionId }

  if (!route) {
    return { engine: 'unavailable', reason: 'route_unavailable', ...location }
  }
  const target = targets.get(peerTargetKey(route))

  if (!target) {
    return { engine: 'checking', reason: 'capabilities_pending', ...location }
  }

  if (!target.ready) {
    return { engine: 'unavailable', reason: target.reason, ...location }
  }

  if (bot.installId && !samePeerInstallation(bot.installId, target.installationId)) {
    return { engine: 'unavailable', reason: 'peer_gateway_changed', ...location }
  }

  return { engine: 'gateway-peer' }
}

/** Roster installId is get_install_id()'s raw id; RoomLink uses install:<id>. */
function samePeerInstallation(installId: string, authorityId: string): boolean {
  return `install:${installId}` === authorityId
}

/** Re-read the complete plan before the first write; callers refresh a rejected plan for another deliberate Create. */
export async function revalidateGroupCreation(
  route: { connectionId: string; profile: string },
  capability: GroupCapabilityState | null,
  source: string,
  selection: RosterRow[],
  expected: GroupCreationEngine,
  unavailable: string
) {
  const sourceCurrent = groupCreationSource(route, capability?.activationEpoch ?? -1)

  if (!capability || capability.source !== source || !sourceCurrent()) {
    throw new Error(unavailable)
  }
  const snapshot = await canonicalGroupRequest<unknown>(route, 'groups.capabilities')

  if (groupExecutionMode(snapshot) !== capability.mode) {
    throw new Error(unavailable)
  }
  const targets = groupSeatsPeers(snapshot)
    ? await readPeerTargets(selection, route.connectionId)
    : new Map<string, PeerTargetState>()
  const plan = groupCreationEngine(groupExecutionMode(snapshot), selection, route.connectionId, snapshot, targets)

  if (!sourceCurrent() || plan.engine !== expected || !['gateway', 'gateway-peer'].includes(plan.engine)) {
    throw new Error(unavailable)
  }

  return { plan, targets, sourceCurrent }
}

/** The name a person knows a bot's connection by. */
export function connectionName(bot: RosterRow | undefined): string {
  return String(bot?.connectionLabel || bot?.connectionId || '')
}

/** The roster rows New Group Chat may seat: never an outage placeholder. Bots on
 *  the current connection must be ones the authority's seatable roster names
 *  (#141178); bots on another connection are offered when the Desktop can reach
 *  them, and their own gateway seats them in the group chat. The ONE predicate
 *  behind both the dialog's picker and the menu item that opens it. */
export function seatableGroupBots(
  roster: RosterRow[],
  {
    connectionId,
    mode,
    seatable
  }: { connectionId: null | string; mode: GroupExecutionMode | 'checking'; seatable: null | Set<string> }
): RosterRow[] {
  return roster.filter(bot => {
    if (bot?.ghost) {
      return false
    }

    if (mode === 'legacy') {
      return true
    }

    if (!botOnConnection(bot, connectionId)) {
      return peerReachable(bot)
    }

    // The profile is the one createCanonicalGroup will send.
    return !seatable || seatable.has(bot.route?.targetProfile ?? bot.targetProfile ?? bot.name)
  })
}

import { gatewayActivationEpoch, host, useGatewayActivationEpoch, useValue } from '@hermes/plugin-sdk'
import { useEffect, useState } from 'react'

import { canonicalGroupRequest } from './canonical-groups'
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

  if (driver === true) {return 'canonical'}

  // App-managed hosted owners can be nonpersistent, even when their driver stops.
  const ownsRooms = Object.hasOwn(capability, 'protocol_version') || Object.hasOwn(capability, 'authority_gateway_id')
    || methods?.includes('groups.create') || features?.includes('canonical_session_owner')

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

  useEffect(() => host.state?.gateway?.listen(() => { if (onForeground(connectionId)) {setOpening(value => value + 1)} }), [connectionId])

  useEffect(() => {
    if (answered || !visible || (onForeground(connectionId) && (host.state?.gateway?.get() ?? 'open') !== 'open')) {return}
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | undefined
    void canonicalGroupRequest<unknown>(route, 'groups.capabilities').then(
      value => { if (!cancelled) {setAnswer({ opening, methods: groupAdvertisedMethods(value) })} },
      () => { if (!cancelled) {timer = setTimeout(() => setAttempt(value => value + 1), 2000)} }
    )

    return () => { cancelled = true; clearTimeout(timer) }
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
export function groupCreationSource(route: { connectionId: string; profile: string },
  activationEpoch = gatewayActivationEpoch()) {
  return () => gatewayActivationEpoch() === activationEpoch &&
    route.connectionId === host.state.connectionId.get() &&
    route.profile === host.state.profile.get() && host.state.gateway.get() === 'open'
}

export interface GroupCapabilityState {
  activationEpoch: number
  mode: GroupExecutionMode
  seatable: null | Set<string>
  source: string
}

/** `groups.capabilities` for the active source while `active`, keyed by that
 *  source so an answer from a source the window has left is never used. */
export function useGroupCapability(active: boolean) {
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

    return () => { cancelled = true }
  }, [activationEpoch, active, connectionId, gateway, profile, source])

  const current = capability?.source === source ? capability : null

  return {
    capability,
    connectionId,
    mode: current ? current.mode : ('checking' as const),
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

/** A bot on another connection the Desktop can carry a room turn to: a live row
 *  whose route resolves to the same connection the row names. */
function courierReachable(bot: RosterRow): boolean {
  if (bot.sourceMissing || bot.sourceReachable === false) {
    return false
  }

  const { route } = resolveBotConnectionRoute(bot)

  return Boolean(route && (!bot.connectionId || route.connectionId === bot.connectionId))
}

function memberConnection(bot: RosterRow, connectionId: null | string): string {
  return String(bot.connectionId || bot.route?.connectionId || connectionId || '').trim()
}

/** True when `members` live on two or more connections. No single gateway can
 *  own such a room, so it runs on this Desktop, which carries each member's turn
 *  to its own gateway. A member with no recorded connection counts as being on
 *  `connectionId`. */
export function groupNeedsDesktopCourier(members: RosterRow[], connectionId: null | string): boolean {
  return new Set(members.map(bot => memberConnection(bot, connectionId))).size > 1
}

/** True when this Desktop drives the room recorded as `room`: the engine it
 *  was created with, else (older records) whether `members` span connections. */
export function groupDrivenByDesktop(room: { engine?: 'desktop' } | undefined, members: RosterRow[], connectionId: null | string): boolean {
  return room?.engine === 'desktop' || groupNeedsDesktopCourier(members, connectionId)
}

/** True when every bot in `selection` lives on ONE connection other than
 *  `connectionId`. The gateway here cannot own that room and no Desktop courier
 *  is needed between machines, so creation from here is refused: the user adds a
 *  bot from this connection or switches to that one. */
export function groupSelectionElsewhere(selection: RosterRow[], connectionId: null | string): boolean {
  return selection.length > 0 && !groupNeedsDesktopCourier(selection, connectionId) &&
    !botOnConnection(selection[0], connectionId)
}

export type GroupCreationEngine = 'desktop' | 'elsewhere' | 'gateway'

/** Who runs a room seating `selection` when created under `mode` from
 *  `connectionId`: the gateway here (canonical mode, every bot on this
 *  connection), this Desktop (a legacy gateway, or bots on two or more
 *  connections), or nobody reachable from here (every bot on ONE other
 *  connection, see groupSelectionElsewhere). */
export function groupCreationEngine(mode: GroupExecutionMode, selection: RosterRow[], connectionId: null | string): GroupCreationEngine {
  if (mode !== 'canonical' || groupNeedsDesktopCourier(selection, connectionId)) {
    return 'desktop'
  }

  return groupSelectionElsewhere(selection, connectionId) ? 'elsewhere' : 'gateway'
}

/** The name a person knows a bot's connection by. */
export function connectionName(bot: RosterRow | undefined): string {
  return String(bot?.connectionLabel || bot?.connectionId || '')
}

/** The roster rows New Group Chat may seat: never an outage placeholder. Bots on
 *  the current connection must be ones the authority's seatable roster names
 *  (#141178); bots on another connection are offered when the Desktop can reach
 *  them, and a selection spanning connections becomes a Desktop-driven room. The
 *  ONE predicate behind both the dialog's picker and the menu item that opens it. */
export function seatableGroupBots(
  roster: RosterRow[],
  { connectionId, mode, seatable }: { connectionId: null | string; mode: GroupExecutionMode | 'checking'; seatable: null | Set<string> }
): RosterRow[] {
  return roster.filter(bot => {
    if (bot?.ghost) {
      return false
    }

    if (mode === 'legacy') {
      return true
    }

    if (!botOnConnection(bot, connectionId)) {
      return courierReachable(bot)
    }

    // The profile is the one createCanonicalGroup will send.
    return !seatable || seatable.has(bot.route?.targetProfile ?? bot.targetProfile ?? bot.name)
  })
}

/**
 * The cross-connection bot relay: the Desktop-as-router loops that let a bot
 * on one gateway reach a bot on another.
 *
 * There is one relay per renderer, so its lifecycle lives in the module-scoped
 * `relay` record below rather than in an atom — no UI reads it. plugin.tsx
 * drives the two doors, startBotRelay / stopBotRelay.
 */

import { host, LruCache } from '@hermes/plugin-sdk'

import { botHandle, clearBotAttention, noteBotAttention } from './data'
import type { ProfileRoute, RosterRow } from './types'

// ── cross-connection bot relay ────────────────────────────────────────────
// Connections ARE the peer set: every gateway this Desktop holds a socket
// to (local, remote URL, SSH, Hermes Cloud, docker) must be able to find
// every other connection's agents and message them via message_agent. The
// Desktop is the relay — it owns every socket. Two loops:
//  - roster loop: pushes each gateway the union roster of agents on the
//    OTHER connections (bot_relay.roster.sync), so message_agent resolves
//    cross-connection targets and Bot Chat prompts list them;
//  - drain loop: collects queued envelopes from every gateway
//    (bot_relay.outbox.drain), delivers each on the target connection's
//    own socket (bot_relay.deliver), and posts the reply back to the
//    sender gateway (bot_relay.reply) where a waiter wakes the sender.
// Older backends without the RPCs fail per-call and are skipped — the
// relay degrades to whatever subset of connections supports it.
const RELAY_ROSTER_INTERVAL_MS = 60_000
// Backstop cadence only (#93594): the push path below carries envelope latency,
// so the interval poll exists for older backends and missed events — 30s
// matches LIVE_SESSION_STATUS_BACKSTOP_INTERVAL_MS. It was 4s back when the
// poll WAS the delivery path, which (before route retention) also meant a
// fresh WebSocket dial + teardown per registered connection every 4s.
const RELAY_DRAIN_INTERVAL_MS = 30_000
// #93911: a delivered turn runs on the target gateway, so the client must
// outlive the backend's own bound. Without this the call fell to the pool's
// generic 30s deadline and every long turn (Computer Use, deep research) came
// back as an unclassified failure.
//
// The backend's MAXIMUM WORK budget is spelled out below. The client deadline
// must be strictly GREATER than it: after those bounded waits the handler still
// has to classify the failure, build and run the retry, classify/serialize the
// terminal result, unwind the temp-file and lock scopes, and get the JSON-RPC
// response back through the event loop. A call that consumes nearly all of the
// work budget would otherwise lose the race to this timer by milliseconds and
// reproduce #93911 at the upper boundary — the backend knowing a typed reason
// while Desktop reports its generic timeout first.
//
// These three are mirrors of backend values, so a change there must not
// silently invalidate this constant: relay-deliver-budget.test.ts reads
// hermes_cli/config_defaults.py and tui_gateway/methods_bot_relay.py and fails
// if the mirrors drift or the margin stops being positive.
const RELAY_TURN_LOCK_WAIT_MS = 120_000 // bot_mode.turn_wait_seconds default
const RELAY_TURN_ATTEMPT_MS = 600_000 // subprocess.run(..., timeout=600)
const RELAY_TURN_MAX_ATTEMPTS = 2 // first attempt + the policy-gated re-run

const RELAY_DELIVER_BACKEND_CEILING_MS = RELAY_TURN_LOCK_WAIT_MS + RELAY_TURN_ATTEMPT_MS * RELAY_TURN_MAX_ATTEMPTS

// Settlement + transport headroom on top of the ceiling, so a backend that
// answers at its own limit still wins the race against this timer.
const RELAY_DELIVER_SETTLEMENT_MARGIN_MS = 180_000
const RELAY_DELIVER_TIMEOUT_MS = RELAY_DELIVER_BACKEND_CEILING_MS + RELAY_DELIVER_SETTLEMENT_MARGIN_MS
// Push path (#93091): the gateway broadcasts `bot_relay.outbox.pending` when
// an envelope lands on disk; a burst of signals inside this window collapses
// to ONE drain. The interval poll above stays as the backstop for older
// backends (and connections whose events don't reach the tap).
const RELAY_PUSH_DEBOUNCE_MS = 250
// A registry route can disappear for a few seconds while its SSH tunnel or
// gateway is restarting. The envelope is already durably claimed by the relay
// at that point, so classifying one stale route snapshot as a permanent
// delivery failure loses an otherwise healthy message. Re-read the registry
// for one bounded gateway-start window before posting a terminal reply.
const RELAY_ROUTE_RECONNECT_GRACE_MS = 30_000
const RELAY_ROUTE_RECONNECT_POLL_MS = 500

/** Everything the two loops mutate. */
interface RelayLifecycle {
  disposed: boolean
  drainBusy: boolean
  /** A push landing while a drain is ALREADY running would be lost forever —
   *  the gateway signature is monotone (one event per new envelope, never
   *  re-broadcast) — so remember it and re-schedule after the drain finishes. */
  drainRerun: boolean
  drainTimer: null | ReturnType<typeof setInterval>
  pushDebounceTimer: null | ReturnType<typeof setTimeout>
  pushUnsub: (() => void) | null
  rosterBusy: boolean
  rosterTimer: null | ReturnType<typeof setInterval>
}

const relay: RelayLifecycle = {
  disposed: false,
  drainBusy: false,
  drainRerun: false,
  drainTimer: null,
  pushDebounceTimer: null,
  pushUnsub: null,
  rosterBusy: false,
  rosterTimer: null
}

// Relay-route socket retention (#93594): connection id → release fn. While
// the relay is active each registered connection's pooled socket is pinned
// open (host.retainProfileSocket) so drain RPCs reuse ONE persistent
// WebSocket instead of dialing and tearing down a fresh one per tick.
// Feature-detected — older shells lack the door and fall back to per-call
// leases. Local routes get a no-op release inside the host (idle-reaper
// exemption). stopBotRelay releases everything.
const relayRouteRetentions = new Map<string, () => void>()
const relayRouteMissDiagnostics = new Map<string, string>()

/** One reachable gateway plus a representative route onto it. The route comes
 *  from `host.profileRoutes()`, which carries identity only — the optional
 *  label fields are read defensively in relayAgentsOn and never arrive. */
interface RelayConnection {
  id: string
  recoveryRelease?: () => void
  /** Registry identity without a materialized renderer route. It may
   *  contribute discovery rows but must never issue RPCs: the SDK's fallback
   *  could otherwise route its `default` profile onto the local gateway. */
  seedOnly?: boolean
  route: ProfileRoute & { connectionLabel?: string; label?: string }
}

/** One agent as pushed to a peer gateway's relay roster. */
interface RelayAgentRow {
  connection_id: string
  connection_label: string
  description: string
  handle: string
  profile: string
  title: string
}

/** A queued cross-connection message drained from a gateway's outbox. */
interface RelayEnvelope {
  id?: string
  idempotency_key?: string
  message?: string
  message_id?: string
  schema?: string
  type?: string
  from_agent?: string
  to_agent?: string
  scope?: Record<string, unknown>
  expires_at?: number
  authority_effect?: string
  target_handle?: string
  target_connection?: string
  target_profile?: string
}

interface RelayTargetReceipt {
  schema: string
  status: string
  idempotency_sha256: string
  message_id: string
  delivery_sha256: string
  target_sha256: string
  target_connection: string
  target_profile: string
  target_handle: string
  started_at: string
  completed_at: string
  reply_sha256: string
}

const RELAY_ENVELOPE_SCHEMA = 'asm-hermes-a2a-envelope/v2'
const RELAY_TARGET_RECEIPT_SCHEMA = 'asm-hermes-a2a-target-receipt/v1'
const RELAY_RECEIPT_READBACK_TIMEOUT_MS = 30_000

class RelayReceiptError extends Error {
  readonly reason: string

  constructor(reason: string, message: string) {
    super(message)
    this.name = 'RelayReceiptError'
    this.reason = reason
  }
}

function isStructuredRelayEnvelope(envelope: RelayEnvelope): boolean {
  return String(envelope?.schema || '') === RELAY_ENVELOPE_SCHEMA
}

async function sha256Hex(value: string): Promise<string> {
  const subtle = globalThis.crypto?.subtle
  if (!subtle) {
    throw new RelayReceiptError(
      'target_receipt_unverified',
      'target receipt reply digest cannot be verified in this runtime'
    )
  }
  const digest = await subtle.digest('SHA-256', new TextEncoder().encode(value))
  return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('')
}

function validateRelayTargetReceipt(
  value: unknown,
  envelope: RelayEnvelope,
  target: RelayConnection
): RelayTargetReceipt {
  if (!value || typeof value !== 'object') {
    throw new RelayReceiptError('target_receipt_unverified', 'target gateway returned no typed receipt')
  }

  const receipt = value as Record<string, unknown>
  const required = [
    'schema', 'status', 'idempotency_sha256', 'message_id', 'delivery_sha256',
    'target_sha256', 'target_connection', 'target_profile', 'target_handle',
    'started_at', 'completed_at', 'reply_sha256'
  ]

  if (required.some(field => typeof receipt[field] !== 'string' || !String(receipt[field]).trim())) {
    throw new RelayReceiptError('target_receipt_unverified', 'target gateway returned an incomplete typed receipt')
  }
  if (receipt.schema !== RELAY_TARGET_RECEIPT_SCHEMA || receipt.status !== 'completed') {
    throw new RelayReceiptError('target_receipt_unverified', 'target gateway receipt is not completed v1')
  }
  if (!/^[0-9a-f]{64}$/.test(String(receipt.idempotency_sha256)) ||
      !/^[0-9a-f]{64}$/.test(String(receipt.delivery_sha256)) ||
      !/^[0-9a-f]{64}$/.test(String(receipt.target_sha256)) ||
      !/^[0-9a-f]{64}$/.test(String(receipt.reply_sha256))) {
    throw new RelayReceiptError('target_receipt_unverified', 'target gateway receipt contains an invalid digest')
  }

  const messageId = String(envelope.message_id || envelope.id || '')
  const targetProfile = String(envelope.target_profile || '')
  const targetHandle = String(envelope.target_handle || envelope.to_agent || '')
  if (receipt.message_id !== messageId ||
      receipt.target_connection !== target.id ||
      receipt.target_profile !== targetProfile ||
      (targetHandle && receipt.target_handle !== targetHandle)) {
    throw new RelayReceiptError('target_receipt_mismatch', 'target gateway receipt identity does not match the envelope route')
  }

  return receipt as unknown as RelayTargetReceipt
}

function assertSameRelayTargetReceipt(left: RelayTargetReceipt, right: RelayTargetReceipt): void {
  const fields: Array<keyof RelayTargetReceipt> = [
    'schema', 'status', 'idempotency_sha256', 'message_id', 'delivery_sha256',
    'target_sha256', 'target_connection', 'target_profile', 'target_handle',
    'started_at', 'completed_at', 'reply_sha256'
  ]
  if (fields.some(field => left[field] !== right[field])) {
    throw new RelayReceiptError('target_receipt_mismatch', 'target receipt changed between delivery and readback')
  }
}

async function assertRelayReplyDigest(
  reply: unknown,
  receipt: RelayTargetReceipt,
  replayed: boolean
): Promise<void> {
  // A replay deliberately returns a suppression notice while the durable
  // receipt retains the original reply digest. The receipt/readback equality
  // check still applies; only a replay skips comparing that notice to the
  // original reply digest.
  if (replayed) {
    return
  }
  if (typeof reply !== 'string') {
    throw new RelayReceiptError('target_receipt_unverified', 'target gateway returned a non-text reply')
  }
  if (await sha256Hex(reply) !== receipt.reply_sha256) {
    throw new RelayReceiptError('target_receipt_mismatch', 'target receipt reply digest does not match the reply')
  }
}

/** Reconcile retention with the CURRENT connection set: pin new connections,
 *  release removed ones. Runs on every drain/roster connection fetch. */
function syncRelayRetention(connections: RelayConnection[]) {
  if (typeof host.retainProfileSocket !== 'function') {
    return
  }

  const live = new Set(connections.map(connection => connection.id))

  for (const [id, release] of [...relayRouteRetentions]) {
    if (!live.has(id)) {
      relayRouteRetentions.delete(id)

      try {
        release()
      } catch {
        // Never let a release failure break the relay loop.
      }
    }
  }

  if (relay.disposed) {
    return
  }

  for (const connection of connections) {
    if (!relayRouteRetentions.has(connection.id)) {
      relayRouteRetentions.set(connection.id, host.retainProfileSocket(connection.route))
    }
  }
}

/** Drop every relay pin — stop/dispose path. */
function releaseRelayRetention() {
  for (const release of relayRouteRetentions.values()) {
    try {
      release()
    } catch {
      // Disposer from an older shell shape — never break teardown.
    }
  }

  relayRouteRetentions.clear()
}

/** One representative route per reachable connection id. */
async function relayConnections(): Promise<RelayConnection[]> {
  if (typeof host.profileRoutes !== 'function' || typeof host.requestProfile !== 'function') {
    return []
  }

  try {
    const routes = await host.profileRoutes()
    const byConnection = new Map<string, ProfileRoute>()

    for (const route of Array.isArray(routes) ? routes : []) {
      const id = String(route?.connectionId || '')

      if (id && !byConnection.has(id)) {
        byConnection.set(id, route)
      }
    }

    return [...byConnection.entries()].map(([id, route]) => ({
      id,
      route
    }))
  } catch {
    return []
  }
}

/** Roster inventory is allowed to seed registered-but-not-yet-materialized
 *  routes. Delivery keeps using relayConnections + waitForRelayConnection so
 *  a seed can never bypass the warm/retention readiness gate. */
async function relayRosterConnections(): Promise<RelayConnection[]> {
  const live = await relayConnections()
  const registeredById = new Map<string, { id: string; kind?: string; remoteProfile?: string }>()

  if (typeof host.connections === 'function') {
    try {
      const registered = await host.connections()

      for (const connection of Array.isArray(registered) ? registered : []) {
        const id = String(connection?.id || '')

        if (id) {
          registeredById.set(id, connection)
        }
      }
    } catch {
      // The union roster below is an independent credential-free registry
      // projection and can carry the peer set through a transient IPC blip.
    }
  }

  if (typeof host.agents === 'function') {
    try {
      const union = await host.agents()

      for (const source of Array.isArray(union?.sources) ? union.sources : []) {
        const id = String(source?.connectionId || '')

        if (id && !registeredById.has(id)) {
          registeredById.set(id, { id, kind: String(source?.kind || '') })
        }
      }
    } catch {
      // Live renderer routes and the primary registry read remain available.
    }
  }

  const byConnection = new Map(live.map(connection => [connection.id, connection]))

  try {
    for (const connection of registeredById.values()) {
      const id = String(connection?.id || '')

      if (!id || byConnection.has(id)) {
        continue
      }

      const kind = String(connection?.kind || '')
      const targetProfile =
        kind === 'ssh' && typeof connection?.remoteProfile === 'string' && connection.remoteProfile.trim()
          ? connection.remoteProfile.trim()
          : 'default'

      const seeded: RelayConnection = {
        id,
        seedOnly: true,
        route: {
          connectionId: id,
          mode: kind === 'local' ? 'local' : 'remote',
          profile: 'default',
          targetProfile
        }
      }

      // Registry membership proves the target is configured, not that its
      // profile socket is ready.  The async retention door performs the same
      // bounded warm/readiness handshake used by delivery recovery; hold its
      // lease through this roster pass so profiles.list cannot race the SSH
      // dashboard startup.
      if (typeof host.warmAgent === 'function') {
        try {
          await host.warmAgent(id, targetProfile)
        } catch {
          // Keep the registered seed in the inventory. The union agent roster
          // below can still supply its last verified rows while this route is
          // reconnecting, and healthy peers must not forget the machine.
        }
      }

      if (typeof host.retainProfile === 'function') {
        try {
          seeded.recoveryRelease = await host.retainProfile(seeded.route)
        } catch {
          // Same fail-soft rule as warmAgent: retain readiness governs direct
          // RPCs, not whether a configured peer exists in roster truth.
        }
      }

      byConnection.set(id, seeded)
    }

    return [...byConnection.values()]
  } catch {
    return [...byConnection.values()]
  }
}

/** Re-acquire one route after a transient registry/tunnel restart.
 *
 * The caller has already claimed the envelope, so this waits in place and
 * never re-enqueues or duplicates it. A genuinely removed connection still
 * receives the existing terminal error after the bounded grace window. */
async function waitForRelayConnection(
  connectionId: string,
  profile: string
): Promise<RelayConnection | undefined> {
  relayRouteMissDiagnostics.delete(connectionId)
  const deadline = Date.now() + RELAY_ROUTE_RECONNECT_GRACE_MS
  let registryIds: string[] = []
  let unionSourceIds: string[] = []

  // profileRoutes is an inventory read; it does not itself re-open a dropped
  // SSH/backend socket. Ask the existing non-foregrounding warm path to dial
  // the target, then poll only for its credential-free route to reappear.
  if (typeof host.warmAgent === 'function') {
    try {
      await host.warmAgent(connectionId, profile)
    } catch {
      // A failed pre-dial can still race with the Desktop's own reconnect.
      // Keep the bounded inventory loop below as the final authority.
    }
  }

  // A remembered SSH roster can momentarily suppress its undialed seed route
  // even though the connection still exists in the authoritative registry.
  // Build the same credential-free descriptor Electron would return so the
  // request path itself can complete the lazy dial. Registry membership is
  // re-read first; arbitrary/stale ids are never synthesized.
  if (typeof host.connections === 'function') {
    try {
      const registered = await host.connections()
      registryIds = (Array.isArray(registered) ? registered : [])
        .map(connection => String(connection?.id || ''))
        .filter(Boolean)

      let source: { id: string; kind?: string; remoteProfile?: string } | undefined = (
        Array.isArray(registered) ? registered : []
      ).find(
        connection => String(connection?.id || '') === connectionId
      )

      // The registry bridge and the union-roster bridge are independent IPC
      // reads. During connection reconciliation one can briefly return an old
      // snapshot while the other already knows the source. The union source
      // contains no endpoint or credential material, but it is enough to
      // prove the exact id is still registered and choose local vs remote.
      if (!source && typeof host.agents === 'function') {
        try {
          const union = await host.agents()
          unionSourceIds = (Array.isArray(union?.sources) ? union.sources : [])
            .map(candidate => String(candidate?.connectionId || ''))
            .filter(Boolean)
          const unionSource = (Array.isArray(union?.sources) ? union.sources : []).find(
            candidate => String(candidate?.connectionId || '') === connectionId
          )

          if (unionSource) {
            source = {
              id: connectionId,
              kind: unionSource.kind
            }
          }
        } catch {
          // The bounded profile-route poll below remains the final fallback.
        }
      }

      if (source) {
        const kind = String(source.kind || '')

        const targetProfile =
          kind === 'ssh' && typeof source.remoteProfile === 'string' && source.remoteProfile.trim()
            ? source.remoteProfile.trim()
            : profile

        const recovered: RelayConnection = {
          id: connectionId,
          route: {
            connectionId,
            mode: kind === 'local' ? 'local' : 'remote',
            profile,
            targetProfile
          }
        }

        // `warmAgent` can lose a race with a gateway restart. The SDK's
        // retained-profile door both dials and waits for readiness without
        // foregrounding the connection. Keep its temporary lease until the
        // standing relay retention below takes ownership of the socket.
        if (typeof host.retainProfile !== 'function') {
          return recovered
        }

        while (!relay.disposed && Date.now() < deadline) {
          try {
            recovered.recoveryRelease = await host.retainProfile(recovered.route)

            return recovered
          } catch {
            await new Promise<void>(resolve => setTimeout(resolve, RELAY_ROUTE_RECONNECT_POLL_MS))
          }
        }

        // Retention is an optimization and a readiness hint, not delivery
        // authority. The registry lookup above already proved this exact
        // connection id still exists. Let the routed request perform one
        // final lazy dial so a renderer-retention race cannot turn a healthy
        // SSH gateway into the misleading "not connected" terminal reply.
        // Any real dial failure is then returned by bot_relay.deliver and is
        // correlated to the claimed envelope instead of being misclassified
        // as missing inventory.
        if (!relay.disposed) {
          return recovered
        }
      }
    } catch {
      // Fall through to bounded route inventory polling.
    }
  }

  while (!relay.disposed && Date.now() < deadline) {
    await new Promise<void>(resolve => setTimeout(resolve, RELAY_ROUTE_RECONNECT_POLL_MS))

    const match = (await relayConnections()).find(connection => connection.id === connectionId)

    if (match) {
      return match
    }
  }

  relayRouteMissDiagnostics.set(
    connectionId,
    `route_unavailable: relay_disposed=${String(relay.disposed)} registry_ids=${registryIds.join(',') || 'none'} union_source_ids=${unionSourceIds.join(',') || 'none'}`
  )

  return undefined
}

/** Rebuild the relay mesh after a cold Desktop launch.
 *
 * The registry persists every gateway, but profileRoutes only includes routes
 * whose backend has been opened in this renderer lifetime. Without this boot
 * step a relaunch silently shrinks the roster to the active connection until
 * the user visits every gateway by hand. Warm each registered source without
 * foregrounding it, wait for the credential-free routes, then resync. */
async function bootstrapRelayConnections() {
  if (typeof host.connections !== 'function' || typeof host.warmAgent !== 'function') {
    return
  }

  try {
    const registered = await host.connections()

    const ids = new Set(
      (Array.isArray(registered) ? registered : [])
        .map(connection => String(connection?.id || ''))
        .filter(Boolean)
    )

    await Promise.all([...ids].map(id => host.warmAgent(id, 'default')))

    const deadline = Date.now() + RELAY_ROUTE_RECONNECT_GRACE_MS

    while (!relay.disposed && Date.now() < deadline) {
      const live = new Set((await relayConnections()).map(connection => connection.id))

      if ([...ids].every(id => live.has(id))) {
        break
      }

      await new Promise<void>(resolve => setTimeout(resolve, RELAY_ROUTE_RECONNECT_POLL_MS))
    }

    if (!relay.disposed) {
      await syncRelayRosters()
    }
  } catch {
    // Older/in-flight registries fall back to the standing roster/drain loops.
  }
}

/** The agents living on one connection, as relay roster rows.
 *  Returns null on FAILURE (transient RPC blip, slow socket) — distinct from
 *  a genuine empty profile list. Conflating the two would push a fresh union
 *  roster missing a LIVE connection's agents, and the gateway-side liveness
 *  check (bot_relay._target_liveness) reads "absent from a fresh roster" as
 *  definitively offline → false runtime_offline refusals (#93091 item 2). */
async function relayAgentsOn(connection: RelayConnection): Promise<RelayAgentRow[] | null> {
  try {
    const res = await host.requestProfile<{ profiles?: RosterRow[] }>(connection.route, 'profiles.list', {
      include_sessions: false
    })

    const profiles = Array.isArray(res?.profiles) ? res.profiles : []
    // TODO(bot-mode-types): neither `connectionLabel` nor `label` can exist on
    // a `host.profileRoutes()` route (connectionId / mode / profile /
    // targetProfile only), so this always falls through to the raw connection
    // id and peer gateways list agents by id instead of the human label.
    const label = String(connection.route?.connectionLabel || connection.route?.label || connection.id)

    return profiles
      .map(profile => ({
        profile: String(profile?.name || ''),
        handle: botHandle(profile?.name, profile),
        connection_id: connection.id,
        connection_label: label,
        title: String(profile?.ui_meta?.['hermes-bots']?.title || profile?.display_name || ''),
        description: String(profile?.description || '')
      }))
      .filter(row => row.profile)
  } catch {
    return null
  }
}

/** Last good agent rows per connection id — reused when a fetch blips so a
 *  transient failure never reads as "everyone on that machine went away".
 *  The sweep below drops disconnected ids, but only on a cycle that had two
 *  or more connections to relay between; the ceiling is what bounds the rest.
 *  Every live connection is rewritten each cycle, so eviction can only ever
 *  reach ids that stopped being fetched. */
const RELAY_AGENTS_CACHE_MAX = 32
const relayAgentsCache = new LruCache<string, RelayAgentRow[]>(RELAY_AGENTS_CACHE_MAX)

/** Push every gateway the union roster of agents on the OTHER connections. */
async function syncRelayRosters() {
  if (relay.disposed || relay.rosterBusy) {
    return
  }

  relay.rosterBusy = true

  let connections: RelayConnection[] = []

  try {
    connections = await relayRosterConnections()

    if (connections.length < 2) {
      return
    }

    const agentsByConnection = new Map<string, RelayAgentRow[]>()
    const unionByConnection = new Map<string, RelayAgentRow[]>()

    // Electron's union registry is the Desktop-wide discovery authority and
    // already preserves registered sources across lazy SSH route churn. Use
    // its thin rows only as a fallback when a gateway's richer profiles.list
    // RPC fails; a genuine empty profile list remains authoritative.
    if (typeof host.agents === 'function') {
      try {
        const union = await host.agents()

        for (const agent of Array.isArray(union?.agents) ? union.agents : []) {
          const id = String(agent?.connectionId || '')
          const profile = String(agent?.targetProfile || agent?.profile || '')

          if (!id || !profile) {
            continue
          }

          const rows = unionByConnection.get(id) || []

          rows.push({
            profile,
            handle: String(agent?.handle || botHandle(profile, { name: profile })),
            connection_id: id,
            connection_label: String(agent?.connectionLabel || id),
            title: '',
            description: ''
          })
          unionByConnection.set(id, rows)
        }
      } catch {
        // Older shells keep using per-gateway discovery only.
      }
    }

    await Promise.all(
      connections.map(async connection => {
        const agents = connection.seedOnly ? null : await relayAgentsOn(connection)

        if (agents === null) {
          // Transient fetch failure: reuse the last good rows for this
          // connection (or contribute nothing this cycle) so the pushed
          // roster never drops a live machine's agents — absence from a
          // fresh roster means offline to the gateway-side fail-fast.
          const cached = relayAgentsCache.get(connection.id) || []
          const unionRows = unionByConnection.get(connection.id) || []
          const declaredProfile = String(connection.route.targetProfile || connection.route.profile || 'default')
          agentsByConnection.set(
            connection.id,
            cached.length > 0
              ? cached
              : unionRows.length > 0
                ? unionRows
                : [
                    {
                      profile: declaredProfile,
                      handle: botHandle(declaredProfile, { name: declaredProfile }),
                      connection_id: connection.id,
                      connection_label: connection.id,
                      title: '',
                      description: ''
                    }
                  ]
          )
        } else {
          relayAgentsCache.set(connection.id, agents)
          agentsByConnection.set(connection.id, agents)
        }
      })
    )

    // Connections gone from profileRoutes are genuinely disconnected — drop
    // their cache so a later reconnect starts from live data.
    const liveIds = new Set(connections.map(connection => connection.id))

    for (const id of [...relayAgentsCache.keys()]) {
      if (!liveIds.has(id)) {
        relayAgentsCache.delete(id)
      }
    }

    await Promise.all(
      connections.map(async connection => {
        if (connection.seedOnly) {
          return
        }

        const others: RelayAgentRow[] = []

        for (const [id, agents] of agentsByConnection) {
          if (id !== connection.id) {
            others.push(...agents)
          }
        }

        try {
          await host.requestProfile(connection.route, 'bot_relay.roster.sync', {
            agents: others
          })
        } catch {
          // Older backend without the relay RPCs — skip this connection.
        }
      })
    )
  } finally {
    for (const connection of connections) {
      try {
        connection.recoveryRelease?.()
      } catch {
        // A temporary readiness lease must never break the standing loop.
      }
    }

    relay.rosterBusy = false
  }
}

/** Drain every gateway's outbox and deliver each envelope on the target
 *  connection's own socket; the reply (or error) is posted back to the
 *  sender gateway for its waiter. */
async function drainRelayOutboxes() {
  if (relay.disposed) {
    return
  }

  if (relay.drainBusy) {
    // A push signal raced an in-flight drain. The gateway never re-sends it
    // (monotone signature), so without this flag the envelope would wait out
    // the full poll interval — exactly the latency the push path removes.
    relay.drainRerun = true

    return
  }

  relay.drainBusy = true

  try {
    // Delivery targets come from the registered roster union, not only from
    // renderer-materialized routes. A connect-on-demand peer is valid target
    // authority even before this renderer has opened its socket; the explicit
    // route descriptor makes requestProfile perform the lazy dial. Seed-only
    // connections are never used as senders below, so no RPC can be
    // misdirected while discovering their outbox.
    const connections = await relayRosterConnections()

    // Retention follows the relay-eligible set: with fewer than two
    // connections there is nothing to relay, so nothing stays pinned.
    syncRelayRetention(connections.length >= 2 ? connections : [])

    if (connections.length < 2) {
      return
    }

    for (const connection of connections) {
      connection.recoveryRelease?.()
      delete connection.recoveryRelease
    }

    const byId = new Map(connections.map(connection => [connection.id, connection]))

    for (const sender of connections) {
      if (sender.seedOnly) {
        continue
      }

      let envelopes: RelayEnvelope[] = []

      try {
        const res = await host.requestProfile<{ envelopes?: RelayEnvelope[] }>(
          sender.route,
          'bot_relay.outbox.drain',
          {}
        )

        envelopes = Array.isArray(res?.envelopes) ? res.envelopes : []
      } catch {
        continue
      }

      for (const envelope of envelopes) {
        if (relay.disposed) {
          return
        }

        const envelopeId = String(envelope?.id || '')
        const targetConnectionId = String(envelope?.target_connection || '')
        let target = byId.get(targetConnectionId)

        const postReply = async (payload: {
          error?: string
          reason?: string
          reply?: string
          target_receipt?: RelayTargetReceipt
        }) => {
          try {
            await host.requestProfile(sender.route, 'bot_relay.reply', {
              id: envelopeId,
              ...payload
            })
          } catch {
            // Sender gateway unreachable — its waiter times out with guidance.
          }
        }

        if (!envelopeId) {
          continue
        }

        if (!target) {
          target = await waitForRelayConnection(
            targetConnectionId,
            String(envelope?.target_profile || 'default')
          )
        }

        if (!target) {
          await postReply({
            error: `connection '${envelope?.target_connection}' is not connected to this Desktop right now`,
            reason: relayRouteMissDiagnostics.get(targetConnectionId) || 'route_unavailable: no diagnostic'
          })

          continue
        }

        // The route returned after this drain's original retention snapshot.
        // Pin it now so delivery and the next loop reuse the recovered socket.
        if (!byId.has(target.id)) {
          byId.set(target.id, target)
          syncRelayRetention([...connections, target])
          target.recoveryRelease?.()
          delete target.recoveryRelease
        }

        // Needs-attention hook (#93091 item 3): a delivered background DM is
        // this bot's "good turn"; a classified delivery failure badges it.
        const attentionKey = `${target.id}::${String(envelope?.target_profile || '')}`

        try {
          const res = await host.requestProfile<{
            reply?: string
            replayed?: boolean
            target_receipt?: unknown
          }>(
            target.route,
            'bot_relay.deliver',
            {
              profile: String(envelope?.target_profile || ''),
              message: String(envelope?.message || ''),
              message_id: String(envelope?.message_id || envelopeId),
              idempotency_key: String(envelope?.idempotency_key || envelopeId),
              envelope_schema: String(envelope?.schema || ''),
              envelope
            },
            RELAY_DELIVER_TIMEOUT_MS
          )

          let targetReceipt: RelayTargetReceipt | undefined
          if (isStructuredRelayEnvelope(envelope)) {
            const deliveredReceipt = validateRelayTargetReceipt(res?.target_receipt, envelope, target)
            await assertRelayReplyDigest(res?.reply, deliveredReceipt, res?.replayed === true)
            const readback = await host.requestProfile<{ receipt?: unknown }>(
              target.route,
              'bot_relay.receipt.read',
              {
                profile: String(envelope?.target_profile || ''),
                message: String(envelope?.message || ''),
                message_id: String(envelope?.message_id || envelopeId),
                idempotency_key: String(envelope?.idempotency_key || envelopeId),
                envelope_schema: RELAY_ENVELOPE_SCHEMA,
                envelope
              },
              RELAY_RECEIPT_READBACK_TIMEOUT_MS
            )
            targetReceipt = validateRelayTargetReceipt(readback?.receipt, envelope, target)
            assertSameRelayTargetReceipt(deliveredReceipt, targetReceipt)
          }

          clearBotAttention(attentionKey)
          await postReply({
            reply: String(res?.reply || ''),
            ...(targetReceipt ? { target_receipt: targetReceipt } : {})
          })
        } catch (error: any) {
          // #93091: bot_relay.deliver classifies the failed turn and ships the
          // typed code in the JSON-RPC error's `data.reason`; forward it into
          // the sender-side reply file so the waiter (and the sending agent)
          // get the machine-readable cause, and prefer it for the badge —
          // classified codes beat free-text re-parsing.
          const reason = String(error?.data?.reason || error?.reason || '').trim()
          noteBotAttention(attentionKey, reason || error?.message || error)
          await postReply({
            error: String(error?.message || error || 'delivery failed'),
            ...(reason
              ? {
                  reason
                }
              : {})
          })
        }
      }
    }
  } finally {
    relay.drainBusy = false

    if (relay.drainRerun && !relay.disposed) {
      // Envelopes signaled mid-drain: schedule one follow-up pass (debounced)
      // instead of leaving them to the interval poll.
      relay.drainRerun = false
      scheduleRelayPushDrain()
    }
  }
}

/** Push-notified drain (#93091): collapse a burst of pending signals into
 *  one drain call ~RELAY_PUSH_DEBOUNCE_MS after the first signal. */
function scheduleRelayPushDrain() {
  if (relay.disposed || typeof setTimeout !== 'function') {
    return
  }

  if (relay.pushDebounceTimer !== null) {
    return
  }

  relay.pushDebounceTimer = setTimeout(() => {
    relay.pushDebounceTimer = null
    void drainRelayOutboxes()
  }, RELAY_PUSH_DEBOUNCE_MS)
}

export function startBotRelay() {
  relay.disposed = false

  // Source-shape test harnesses evaluate plugin.js without DOM timers —
  // the relay only runs where a real event loop exists.
  if (typeof setInterval !== 'function' || typeof clearInterval !== 'function') {
    return
  }

  if (relay.rosterTimer === null) {
    relay.rosterTimer = setInterval(() => void syncRelayRosters(), RELAY_ROSTER_INTERVAL_MS)
    void syncRelayRosters()
    void bootstrapRelayConnections()
  }

  if (relay.drainTimer === null) {
    relay.drainTimer = setInterval(() => void drainRelayOutboxes(), RELAY_DRAIN_INTERVAL_MS)
  }

  // Push path: the gateway change watcher broadcasts when an envelope hits
  // the outbox; drain immediately (debounced) instead of waiting the poll
  // out. Feature-detected — older shells have no host.onEvent — and the 4s
  // poll above stays untouched as the backstop either way.
  if (relay.pushUnsub === null && typeof host.onEvent === 'function') {
    relay.pushUnsub = host.onEvent('bot_relay.outbox.pending', () => scheduleRelayPushDrain())
  }
}

export function stopBotRelay() {
  relay.disposed = true
  // A rerun remembered mid-drain must not leak into the next start —
  // it would fire one stale drain after restart.
  relay.drainRerun = false
  // Unpin every relay-retained socket (#93594): with the relay stopped the
  // pooled entries return to dispose-at-refcount-0 semantics.
  releaseRelayRetention()

  if (relay.rosterTimer !== null) {
    clearInterval(relay.rosterTimer)
    relay.rosterTimer = null
  }

  if (relay.drainTimer !== null) {
    clearInterval(relay.drainTimer)
    relay.drainTimer = null
  }

  if (relay.pushDebounceTimer !== null) {
    clearTimeout(relay.pushDebounceTimer)
    relay.pushDebounceTimer = null
  }

  if (relay.pushUnsub !== null) {
    try {
      relay.pushUnsub()
    } catch {
      // Disposer from an older shell shape — never break teardown.
    }

    relay.pushUnsub = null
  }
}

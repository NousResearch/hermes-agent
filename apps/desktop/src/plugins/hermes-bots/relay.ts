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
// outlive the backend's own bound — worst case is the turn lock wait
// (bot_mode.turn_wait_seconds, default 120s) plus a 600s turn, doubled when
// the retry policy grants one bounded re-run (methods_bot_relay.py). Without
// this the call fell to the pool's generic 30s deadline and every long turn
// (Computer Use, deep research) came back as an unclassified failure.
const RELAY_DELIVER_TIMEOUT_MS = 1_320_000
// Push path (#93091): the gateway broadcasts `bot_relay.outbox.pending` when
// an envelope lands on disk; a burst of signals inside this window collapses
// to ONE drain. The interval poll above stays as the backstop for older
// backends (and connections whose events don't reach the tap).
const RELAY_PUSH_DEBOUNCE_MS = 250

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

/** One reachable gateway plus a representative route onto it. The route comes
 *  from `host.profileRoutes()`, which carries identity only — the optional
 *  label fields are read defensively in relayAgentsOn and never arrive. */
interface RelayConnection {
  id: string
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
  message?: string
  target_connection?: string
  target_profile?: string
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

  try {
    const connections = await relayConnections()

    if (connections.length < 2) {
      return
    }

    const agentsByConnection = new Map<string, RelayAgentRow[]>()
    await Promise.all(
      connections.map(async connection => {
        const agents = await relayAgentsOn(connection)

        if (agents === null) {
          // Transient fetch failure: reuse the last good rows for this
          // connection (or contribute nothing this cycle) so the pushed
          // roster never drops a live machine's agents — absence from a
          // fresh roster means offline to the gateway-side fail-fast.
          agentsByConnection.set(connection.id, relayAgentsCache.get(connection.id) || [])
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
    const connections = await relayConnections()

    // Retention follows the relay-eligible set: with fewer than two
    // connections there is nothing to relay, so nothing stays pinned.
    syncRelayRetention(connections.length >= 2 ? connections : [])

    if (connections.length < 2) {
      return
    }

    const byId = new Map(connections.map(connection => [connection.id, connection]))

    for (const sender of connections) {
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
        const target = byId.get(String(envelope?.target_connection || ''))

        const postReply = async (payload: { error?: string; reason?: string; reply?: string }) => {
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
          await postReply({
            error: `connection '${envelope?.target_connection}' is not connected to this Desktop right now`
          })

          continue
        }

        // Needs-attention hook (#93091 item 3): a delivered background DM is
        // this bot's "good turn"; a classified delivery failure badges it.
        const attentionKey = `${target.id}::${String(envelope?.target_profile || '')}`

        try {
          const res = await host.requestProfile<{ reply?: string }>(
            target.route,
            'bot_relay.deliver',
            {
              profile: String(envelope?.target_profile || ''),
              message: String(envelope?.message || '')
            },
            RELAY_DELIVER_TIMEOUT_MS
          )

          clearBotAttention(attentionKey)
          await postReply({
            reply: String(res?.reply || '')
          })
        } catch (error: any) {
          // #93091: bot_relay.deliver classifies the failed turn and ships the
          // typed code in the JSON-RPC error's `data.reason`; forward it into
          // the sender-side reply file so the waiter (and the sending agent)
          // get the machine-readable cause, and prefer it for the badge —
          // classified codes beat free-text re-parsing.
          const reason = String(error?.data?.reason || '').trim()
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

/**
 * ONE connection supervisor per window (#C5).
 *
 * Before this module, one window's reconnection state machine had three owners,
 * each with its own trigger and its own ticket:
 *
 *  * the PRIMARY ladder — `use-gateway-boot` held a `setTimeout` and an attempt
 *    counter and re-dialed the window's socket on its own clock;
 *  * the POOL ladder — `store/gateway` held a `setTimeout` and an attempt
 *    counter per pooled entry;
 *  * the WAKE path — `store/gateway-reconnect` held a module-level single-flight
 *    over one registered handler, and `use-gateway-boot-wake` nudged the whole
 *    pool on every signal.
 *
 * A wake therefore fired three redial families, each re-minting its own tickets:
 * measured, up to nine sockets opened inside one five-second burst.
 *
 * This module is the one owner of the four things those three needed:
 *
 *  1. the single in-flight dial per route (`dialRoute`) — a route already
 *     dialing hands its in-flight promise to the next caller instead of
 *     opening a second socket;
 *  2. the backoff ladder, one per route (`armRouteRetry`) — one implementation
 *     of the full-jitter exponential shape instead of two;
 *  3. the epoch counter (`bumpGatewayActivationEpoch`) — the supervisor is its
 *     sole writer, so "the active route moved" and "the socket was rebuilt"
 *     are stamped by one owner;
 *  4. wake-signal handling (`superviseWake`) — ONE coalesced action per signal
 *     burst, with the destructive half behind its own holdoff.
 *
 * It is a leaf: it imports the shared backoff shape, the activation-epoch atom
 * and the parked-while-hidden latch, and nothing here reaches back into the
 * registry, the boot hook or the pool. The window's owners hand their work in as
 * ports (dependency inversion): `use-gateway-boot` registers the primary route,
 * the wake module registers the pool route.
 *
 * The primary is the supervisor's first-class route: a wake dials it through
 * `dialPrimaryRoute`, and the pool is only ever nudged from inside that same
 * single action — pooled routes never re-open on a wake timer of their own.
 */
import { reconnectBackoffDelayMs } from '@hermes/shared'

import { $gatewayActivationEpoch } from '@/store/gateway-activation-epoch'
import { secondaryReconnectsParked } from '@/store/gateway-park'

/** The route id of the window's primary socket, in the supervisor's dial and ladder ledgers. */
export const PRIMARY_ROUTE = 'primary'

/**
 * The route id of one pooled (secondary) socket. Scopes and the primary id share
 * the ledgers, so the prefix keeps a profile literally named "primary" from
 * aliasing the window's socket.
 */
export function secondaryRouteId(scope: string): string {
  return `scope:${scope}`
}

/**
 * Why a reconnect is running. `'manual'` (the default) is an explicit user
 * recovery — the handler may unconditionally re-dial, including retrying a
 * credential that requires sign-in. `'restart-followthrough'` is the automatic
 * hand-off after a gateway restart: the socket often SURVIVED (the restart
 * targets the messaging gateway, not this client's backend), so the handler
 * probes first instead of force-closing a healthy socket. `'wake'` is a
 * window-lifecycle recovery signal (visibility/focus/online/power-resume): the
 * socket after a sleep is presumptively dead, so the handler DIALS a replacement
 * immediately — no liveness probe — and retires the old generation only once the
 * replacement reports ready (#C2a).
 */
export type GatewayReconnectSource = 'manual' | 'restart-followthrough' | 'wake'

export interface GatewayReconnectOptions {
  source?: GatewayReconnectSource
}

export type GatewayReconnectHandler = (options?: GatewayReconnectOptions) => Promise<void> | void

/** The window's primary socket, as the boot hook supplies it. */
export interface GatewayPrimaryRoute {
  dial: GatewayReconnectHandler
  /**
   * Does the window's primary socket REPORT itself open right now? Omitted when
   * the route cannot say — the supervisor then treats it as "not open", so a
   * soft signal still dials.
   *
   * A renderer WebSocket object keeps `readyState === OPEN` while the OS threw
   * the transport away, so this is a report, not a proof. It decides one thing
   * only: a SOFT signal (a plain focus) on a socket that reports open dials
   * nothing — an alt-tab says nothing about the socket, and replacing a healthy
   * one per tab-switch is the #94769 churn class on the primary. A HARD signal
   * replaces the socket either way, and the wake module's own contract keeps a
   * hidden window from reaching this at all.
   */
  reportsOpen?: () => boolean
}

/** The window's pooled sockets, as the wake module supplies them. */
export interface GatewayPoolRoute {
  /**
   * Nudge every pooled route: re-arm each entry and redial the ones that are
   * already closed. With `forceOpenSockets` it also probes/closes the ones that
   * report open — the DESTRUCTIVE half, which the supervisor gates behind
   * `WAKE_SWEEP_HOLDOFF_MS` (closing an open socket detaches its runtime, so it
   * must never ride every signal — the #94769 flicker loop).
   */
  nudge: (options: { forceOpenSockets: boolean }) => void
}

interface RouteLadder {
  attempt: number
  timer: ReturnType<typeof setTimeout> | null
}

interface SupervisorState {
  /** The window's primary route, `null` before the boot hook mounts. */
  primary: GatewayPrimaryRoute | null
  /** The window's pool route, `null` before the wake module is created. */
  pool: GatewayPoolRoute | null
  /** In-flight dial per route id; the promise concurrent callers join. */
  dials: Map<string, Promise<void>>
  /** The backoff ladder per route id. */
  ladders: Map<string, RouteLadder>
  /** The wake action in flight, if any: every signal that arrives during it joins it. */
  wakeAction: Promise<void> | null
  /** Wall-clock of the last DESTRUCTIVE pool sweep (see WAKE_SWEEP_HOLDOFF_MS). */
  lastForcedSweepAt: number
}

const STATE_KEY = Symbol.for('hermes.desktop.gatewaySupervisor')

function createSupervisorState(): SupervisorState {
  return {
    primary: null,
    pool: null,
    dials: new Map<string, Promise<void>>(),
    ladders: new Map<string, RouteLadder>(),
    wakeAction: null,
    lastForcedSweepAt: 0
  }
}

let plainState: SupervisorState | null = null

// Same dev-HMR rule as the gateway registry: this module is imported by the
// registry, the boot hook and the reconnect façade, so an edit anywhere in that
// fan-out must hand back the SAME ledgers instead of orphaning a live window's
// arming timers and registrations. Production strips `import.meta.hot` and gets
// a plain module-local container.
function supervisorState(): SupervisorState {
  if (import.meta.hot) {
    const store = globalThis as unknown as { [STATE_KEY]?: SupervisorState }
    store[STATE_KEY] ??= createSupervisorState()

    return store[STATE_KEY]
  }

  return (plainState ??= createSupervisorState())
}

const s = (): SupervisorState => supervisorState()

// ── the epoch counter ───────────────────────────────────────────────────────

/**
 * Advance the gateway activation epoch and return the new value. The supervisor
 * is its only writer: every active-route selection in `store/gateway.ts` bumps
 * it, and a selection that returns a stale epoch is fenced out. A consumer that
 * must re-read its route after an activation reads it through
 * `useGatewayActivationEpoch` / `gatewayActivationEpoch`, not from here.
 */
export function bumpGatewayActivationEpoch(): number {
  $gatewayActivationEpoch.set($gatewayActivationEpoch.get() + 1)

  return $gatewayActivationEpoch.get()
}

// ── the single in-flight dial, one per route ────────────────────────────────

/**
 * Run `run` as its route's ONE in-flight dial. A caller that arrives while the
 * route is already dialing joins that dial (the same promise) instead of opening
 * a second socket — this is what makes "zero duplicate concurrent dials to the
 * same route" structural rather than a property of each caller's guards.
 */
export function dialRoute(routeId: string, run: () => Promise<void> | void): Promise<void> {
  const inFlight = s().dials.get(routeId)

  if (inFlight) {
    return inFlight
  }

  // Start on a microtask, exactly as the reconnect façade always has: the
  // booking below must be synchronous for two same-tick callers to join, while
  // the dial itself stays out of the caller's frame.
  const dialing = Promise.resolve()
    .then(() => run())
    .finally(() => {
      if (s().dials.get(routeId) === dialing) {
        s().dials.delete(routeId)
      }
    })

  s().dials.set(routeId, dialing)

  return dialing
}

/** Dial the window's primary route. Rejects when no boot owner is mounted. */
export function dialPrimaryRoute(options?: GatewayReconnectOptions): Promise<void> {
  const route = s().primary

  if (!route) {
    return Promise.reject(new Error('Gateway reconnect is unavailable'))
  }

  return dialRoute(PRIMARY_ROUTE, () => route.dial(options))
}

/** True while `routeId` has a dial in flight. */
export function routeDialInFlight(routeId: string): boolean {
  return s().dials.has(routeId)
}

// ── the backoff ladder, one per route ───────────────────────────────────────

function ladderFor(routeId: string): RouteLadder {
  const ladders = s().ladders
  let ladder = ladders.get(routeId)

  if (!ladder) {
    ladder = { attempt: 0, timer: null }
    ladders.set(routeId, ladder)
  }

  return ladder
}

/**
 * Arm `routeId`'s next retry on the supervisor's own ladder: full-jitter
 * exponential backoff (300 ms base, 15 s cap) so a gateway restart is not
 * redialed by every Desktop window in lockstep. There is ONE timer per route:
 * a second arm while one is pending is refused, so no caller can stack ladders.
 *
 * The armed run is skipped when the window is parked (#C2a): a hidden window
 * dials nothing, whatever was armed before the park went up.
 */
export function armRouteRetry(routeId: string, run: () => void): void {
  const ladder = ladderFor(routeId)

  if (ladder.timer !== null) {
    return
  }

  const delay = reconnectBackoffDelayMs(ladder.attempt)

  ladder.attempt += 1
  ladder.timer = setTimeout(() => {
    ladder.timer = null

    if (secondaryReconnectsParked()) {
      return
    }

    run()
  }, delay)
}

/** Disarm `routeId`'s pending retry, keeping its ladder position. */
export function cancelRouteRetry(routeId: string): void {
  const ladder = s().ladders.get(routeId)

  if (ladder?.timer != null) {
    clearTimeout(ladder.timer)
    ladder.timer = null
  }
}

/** True while `routeId` has an armed retry. */
export function routeRetryPending(routeId: string): boolean {
  return s().ladders.get(routeId)?.timer != null
}

/** Back to the first rung: the route opened healthy (or is starting over). */
export function resetRouteLadder(routeId: string): void {
  const ladder = s().ladders.get(routeId)

  if (ladder) {
    ladder.attempt = 0
  }
}

/** Cancel a route's retry AND forget its ladder position (the route is gone). */
export function forgetRoute(routeId: string): void {
  cancelRouteRetry(routeId)
  s().ladders.delete(routeId)
}

// ── the window's routes ─────────────────────────────────────────────────────

export function registerGatewayPrimaryRoute(route: GatewayPrimaryRoute): () => void {
  s().primary = route

  return () => {
    if (s().primary === route) {
      s().primary = null
    }
  }
}

export function registerGatewayPoolRoute(route: GatewayPoolRoute): () => void {
  s().pool = route

  return () => {
    if (s().pool === route) {
      s().pool = null
    }
  }
}

// ── wake-signal handling ────────────────────────────────────────────────────

/** How long the DESTRUCTIVE half of a wake (closing pooled sockets that report
 *  open) stays quiet after one runs. Destruction is the re-home, so it is the
 *  expensive half; the cheap half — dial the primary, redial the pooled routes
 *  that are already closed — is idempotent and needs no holdoff. */
export const WAKE_SWEEP_HOLDOFF_MS = 15_000

/**
 * Take ONE wake signal: the cheap path (nudge the pool, dial the primary) plus,
 * on a HARD signal and no more than once per holdoff, the destructive sweep.
 *
 * A signal that arrives while the action is running JOINS it. That is the whole
 * fix: a real resume fires several signals (power resume, network online, the
 * window becoming visible) seconds apart, and each one used to start its own
 * redial family. They now land on one action, one ticket batch, one dial.
 *
 * `hard` = the route's liveness is genuinely in question: a network that came
 * back, or the return from a park. A SOFT signal (a plain focus) is not: the
 * caller decides, and `use-gateway-boot-wake` keeps an alt-tab from churning a
 * healthy socket.
 */
export function superviseWake(hard: boolean): void {
  const state = s()

  if (secondaryReconnectsParked() || state.wakeAction) {
    return
  }

  const action = runWakeAction(hard).catch(() => undefined)

  state.wakeAction = action
  void action.finally(() => {
    if (state.wakeAction === action) {
      state.wakeAction = null
    }
  })
}

/** True while a wake action is running. */
export function gatewayWakeInFlight(): boolean {
  return s().wakeAction !== null
}

async function runWakeAction(hard: boolean): Promise<void> {
  const state = s()
  const now = Date.now()
  const forced = hard && now - state.lastForcedSweepAt >= WAKE_SWEEP_HOLDOFF_MS

  if (forced) {
    state.lastForcedSweepAt = now
  }

  // Pool first, primary second — the order the wake path has always used, so a
  // socket the pool is about to re-dial is queued before the window's own dial
  // starts competing for the same backend.
  state.pool?.nudge({ forceOpenSockets: forced })

  if (!state.primary) {
    return
  }

  // A soft signal says nothing about the primary's transport: on a socket that
  // reports open it dials nothing. Hard always dials — the socket after a sleep
  // is presumptively dead whatever it reports.
  if (!hard && state.primary.reportsOpen?.() === true) {
    return
  }

  // Single-flighted by `dialRoute`: a dial already in flight (a wake that
  // arrived while the previous one was still connecting) is joined.
  await dialPrimaryRoute({ source: 'wake' })
}

// ── lifecycle ───────────────────────────────────────────────────────────────

/**
 * Test-only: forget every registration, dial, ladder and holdoff stamp. Module
 * state outlives one test file's cases the same way it outlives one HMR pass.
 */
export function resetGatewaySupervisorForTests(): void {
  for (const ladder of s().ladders.values()) {
    if (ladder.timer !== null) {
      clearTimeout(ladder.timer)
      ladder.timer = null
    }
  }

  const state = createSupervisorState()

  Object.assign(s(), state)
}

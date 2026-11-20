/**
 * Wake recovery for one window's primary gateway socket (#C2a).
 *
 * Extracted from the useGatewayBoot effect: it is the one part of that hook
 * with state of its own (a park latch, a holdoff stamp, the sockets a wake kept
 * alive), and the effect is over its length cap. The host supplies the window's
 * primary and the four things only the hook can do (route resolution,
 * attach/promote, post-connect work, reauth surfacing); everything decided here
 * is decided from renderer-wide stores.
 *
 * Two behaviours live here:
 *
 * 1. PARK. On `visibilitychange → hidden` the window stops paying reconnect
 *    costs: the primary ladder and every pool ladder are disarmed and latched,
 *    and the routes are STALE from then on — the app stops trusting that any
 *    socket survived the time the user was away. Nothing is torn down: the
 *    cached transcript keeps painting and a genuinely alive socket stays up.
 *    A hidden window dials NOTHING, whatever signal arrives (an interface
 *    change, a lid resume that leaves the app backgrounded), which is what
 *    makes a backgrounded app free and a ten-minute background mint zero
 *    tickets.
 *
 * 2. DIAL, DON'T PROBE. On the return — and on online / power-resume — a
 *    socket that still reports OPEN is presumptively dead: the renderer's
 *    WebSocket object keeps `readyState === OPEN` while the OS threw the
 *    transport away, and no close event ever fires. The old path PROBED that
 *    socket with a 5 s ping before deciding, which is strictly slower than the
 *    dial it protects (measured ticket→accept p50 0.02-0.07 s on the test
 *    instance) and left the user watching a dead socket be proven dead. A hard
 *    wake now MINTS AND DIALS a replacement generation at once — while the old
 *    socket stays open, so the window is never un-owned — and retires the old
 *    generation at the earliest safe point:
 *      * nothing was riding it → retire it as soon as the replacement is open;
 *      * a turn was riding it  → keep it (that is #95327: a busy backend must
 *        never have a socket torn out from under a running turn), keep it
 *        SERVED — its listeners live exactly as long as its socket, so the
 *        running turn's deltas on it still reach the app — and retire it once
 *        that turn's claim drains.
 *    The replacement INHERITS the old generation's replay cursors before it
 *    dials: a fresh client that never asks `session.events.since` is a cold
 *    boot, and every event of the half-open window would be lost.
 *    A plain focus is not a hard signal: an alt-tab must never swap a socket
 *    that reports healthy (the #94769 churn class on the primary); it redials
 *    only a primary that is not open.
 *    The liveness probe keeps exactly that #95327 job on the paths that still
 *    need it: the restart-followthrough and the pool's forced sweep.
 */
import type { HermesConnection } from '@/global'
import { HermesGateway } from '@/hermes'
import { shouldRetireReplacedSocket } from '@/lib/gateway-liveness-policy'
import { ensureReconnectEpisode, noteReconnectDetection, settleReconnectEpisode } from '@/store/desktop-metrics'
import { reconnectSecondaryGateways } from '@/store/gateway'
import {
  parkSecondaryReconnects,
  secondaryReconnectsParked,
  unparkSecondaryReconnects
} from '@/store/gateway-park'
import { registerGatewayPoolRoute, superviseWake } from '@/store/gateway-supervisor'
import { reconnectResumeGate } from '@/store/reconnect-resume'
import { $workingSessionIds } from '@/store/session-states'

export interface WakeRoute {
  conn: HermesConnection
  wsUrl: string
}

export interface WakeHost {
  /** True while a recovery step may proceed: booted, not switching, not torn down. */
  ready: () => boolean
  /** The window's CURRENT primary client. */
  primary: () => HermesGateway
  /** Subscribe a new generation to the window's primary handlers; returns its detach. */
  attach: (client: HermesGateway) => () => void
  /**
   * Make `client` this window's primary: bind it, republish, announce it.
   * Returns the REPLACED generation's detach — the wake module owns it, and
   * runs it the moment that generation closes (listener lifetime equals
   * socket lifetime: a kept replacement is kept SERVED, #95327).
   */
  promote: (client: HermesGateway, detach: () => void, conn: HermesConnection) => () => void
  /** Resolve the primary route and mint its ticket; null when a newer route owns the socket. */
  resolveRoute: () => Promise<null | WakeRoute>
  /** Dial in place, for a signal with no socket to protect. */
  redial: () => Promise<void>
  /** Post-connect work for a replacement generation (no respawn reconcile). */
  finish: (conn: HermesConnection) => Promise<void>
  /** Surface a stale credential once; every other failure leaves the live socket alone. */
  noteReauth: (error: unknown) => void
}

export interface WakeRecovery {
  /**
   * Wire the window's recovery signals to `signal` (the caller's "the primary
   * ladder should stand down and dial again" step), which takes the signal's
   * class (hard = the route's liveness is genuinely in question). Visibility
   * changes park and unpark on the way through; returns one detach.
   */
  bindSignals: (signal: (hard: boolean) => void) => () => void
  /** Retire any replaced generation whose in-flight work has drained. */
  drain: () => void
  /** Close every socket this module kept alive and drop its subscriptions. */
  dispose: () => void
  /** Park: disarmed ladders, stale routes, nothing torn down. */
  park: () => void
  /** Return: the routes are live again. Never dials by itself — the caller wakes. */
  unpark: () => void
  /**
   * One wake signal: sweep the pool once per holdoff, and dial the primary
   * once (single-flighted). A SOFT signal (plain focus) leaves a primary that
   * reports open alone — alt-tab must not churn a healthy socket; a HARD
   * signal (return-from-park / online / power-resume) replaces one.
   *
   * The action itself lives in the window's connection supervisor
   * (store/gateway-supervisor.superviseWake): a signal that arrives while one is
   * running joins it, so a burst of wake signals is one ticket batch and one
   * dial rather than one per signal per family.
   */
  wake: (hard: boolean) => void
  /** The `'wake'` reconnect handler: dial a replacement generation. */
  dial: () => Promise<void>
}

export function createWakeRecovery(host: WakeHost): WakeRecovery {
  /**
   * Sockets a wake replaced but kept, because a turn was riding them (#95327).
   * Each entry carries the generation's app-listener detach: a kept socket is
   * a SERVED socket — the running turn's deltas on it still reach the app,
   * which is the whole point of keeping it — and the detach runs in the same
   * breath as the close, never before.
   */
  const replacedGenerations: Array<{ client: HermesGateway; detach: () => void }> = []

  // The pool's half of a wake, handed to the window's supervisor: a signal
  // nudges the pooled routes from inside the supervisor's ONE coalesced action
  // (and only from there), so a burst of signals cannot start one redial family
  // per signal. The destructive half is the supervisor's holdoff to gate.
  const offPoolRoute = registerGatewayPoolRoute({ nudge: reconnectSecondaryGateways })

  const closeReplaced = () => {
    for (const previous of replacedGenerations.splice(0)) {
      previous.detach()

      try {
        previous.client.close()
      } catch {
        // Already torn down by the backend or the OS.
      }
    }
  }

  const drain = () => {
    if (replacedGenerations.length === 0 || !shouldRetireReplacedSocket($workingSessionIds.get().length)) {
      return
    }

    closeReplaced()
  }

  // A replaced socket that was kept for a turn is retired the moment that work
  // drains (#95327): the module owns the trigger, so the window does not have
  // to remember to call drain() from the right subscription.
  const offWorkDrained = $workingSessionIds.subscribe(drain)

  /**
   * Retire a replaced socket now, or keep it (SERVED) until its work drains
   * (#95327). The in-flight-work signal is read HERE, at the signal boundary:
   * the dial that follows reconciles runtime bindings for the replaced socket,
   * and that reconcile is not the turn ending.
   */
  const retireReplaced = (previous: HermesGateway, detachPrevious: () => void) => {
    if (!shouldRetireReplacedSocket($workingSessionIds.get().length)) {
      replacedGenerations.push({ client: previous, detach: detachPrevious })

      return
    }

    detachPrevious()
    previous.close()
  }

  /**
   * Park, and end any wait the user was in the middle of. A close that landed
   * while the window was VISIBLE opened the episode its user is waiting on; a
   * park that beats that redial means the wait is over — nobody is waiting any
   * more, and the parked time must never land inside the next episode's clock.
   * The return's dial redials what the park left behind (a primary nobody
   * watched die), which the next episode reports as `parked_close`.
   */
  const park = () => {
    parkSecondaryReconnects()
    settleReconnectEpisode('parked')
  }

  const dial = async (): Promise<void> => {
    if (!host.ready() || secondaryReconnectsParked()) {
      return
    }

    const previous = host.primary()

    if (previous.connectionState !== 'open') {
      // Nothing to protect and nothing to replace: the ordinary dial. A window
      // parked while its socket died comes back through exactly this branch.
      await host.redial()

      return
    }

    // C1 episode: a wake fires no close edge, so the signal starts the clock and
    // the redial decision below ends detection.
    ensureReconnectEpisode({ closeReason: 'wake', visibilityAtClose: windowVisibility() })
    noteReconnectDetection()

    // The episode opened above describes THIS dial's replacement. Up to the
    // promote, the live primary is still the one this wake meant to replace, so
    // an abandoned dial here must end its own episode — left open, the next
    // unrelated usable/begin would settle it on a clock that is not its own.
    let promoted = false

    const abandonEpisode = () => {
      if (!promoted && host.primary().connectionState === 'open') {
        settleReconnectEpisode('abandoned')
      }
    }

    try {
      const route = await host.resolveRoute()

      if (!route) {
        // No route this dial may touch (its contract: null when a newer
        // boot/apply owns the socket, or the window is going away). The episode
        // above is THIS dial's — end it on its own clock like every other exit
        // here; left open, the next unrelated usable/begin would settle it.
        abandonEpisode()

        return
      }

      const successor = new HermesGateway({ beforeOpenRequests: reconnectResumeGate })
      // A CONTINUATION, not a cold boot (#C2a r2): share the replay cursors
      // so this dial asks `session.events.since` for the window the old,
      // half-open socket could not deliver. Must run before connect() — the
      // open-time fetchReplay reads them.
      successor.inheritReplayState(previous)
      const detach = host.attach(successor)

      try {
        await successor.connect(route.wsUrl)
      } catch (error) {
        detach()
        successor.close()
        throw error
      }

      if (!host.ready() || successor.connectionState !== 'open') {
        detach()
        successor.close()
        abandonEpisode()

        return
      }

      const detachPrevious = host.promote(successor, detach, route.conn)

      promoted = true
      await host.finish(route.conn)
      retireReplaced(previous, detachPrevious)
    } catch (error) {
      // A stale ticket still has to surface once (it can never succeed); a
      // transport failure is not a lost connection — the live socket stays.
      host.noteReauth(error)
      abandonEpisode()
    } finally {
      drain()
    }
  }

  return {
    bindSignals: (signal: (hard: boolean) => void) => {
      // A fresh bind RE-READS the document: the latch is a function of what the
      // window is doing, so a stale park (an editor reload while hidden, a mount
      // that missed the event) can never latch a visible window's ladders off.
      if (document.visibilityState === 'hidden') {
        park()
      } else {
        unparkSecondaryReconnects()
      }

      // HARD = the route's liveness is genuinely in question: a network that
      // came back, or the return from a park (becoming visible IS coming back
      // from hidden — the latch went up on the way out). SOFT = the window
      // merely regained focus: an alt-tab says nothing about the socket, and
      // replacing an open one on it is a mint + churn per tab-switch (#94769
      // on the primary) — a soft signal redials only a non-open primary.
      const onOnline = () => signal(true)
      const onFocus = () => signal(false)

      const onVisible = () => {
        if (document.visibilityState === 'visible') {
          unparkSecondaryReconnects()
          signal(true)

          return
        }

        park()
      }

      window.addEventListener('online', onOnline)
      window.addEventListener('focus', onFocus)
      document.addEventListener('visibilitychange', onVisible)

      return () => {
        window.removeEventListener('online', onOnline)
        window.removeEventListener('focus', onFocus)
        document.removeEventListener('visibilitychange', onVisible)
      }
    },
    dial,
    drain,
    dispose: () => {
      offWorkDrained()
      offPoolRoute()
      closeReplaced()
    },
    park,
    unpark: unparkSecondaryReconnects,
    wake: (hard: boolean) => {
      if (!host.ready() || secondaryReconnectsParked()) {
        return
      }

      // ONE action per signal, in the window's supervisor: it nudges the pool,
      // dials the primary once (single-flight — a burst of wake signals joins
      // the dial already in flight) and gates the destructive sweep behind its
      // own holdoff. A focus on a socket that reports open dials nothing; the
      // supervisor reads that from the primary's own report.
      superviseWake(hard)
    }
  }
}

function windowVisibility(): 'hidden' | 'visible' {
  return document.visibilityState === 'hidden' ? 'hidden' : 'visible'
}

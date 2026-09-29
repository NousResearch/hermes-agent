import { resolveRendererBootWaitMs } from '../../../shared/src/desktop-boot-budget'

/** Budget for the renderer awaits that can be waiting on a backend cold
 * spawn's whole chain — boot-class callers only. Three sites, all named
 * here so a new caller must decide which class it is:
 *   1. initial boot()'s getConnection() (use-gateway-boot.ts) — the primary
 *      cold spawn this is sized for;
 *   2. a profile/connection switch whose ensureBackend may cold-spawn a
 *      pooled helper (use-gateway-boot.ts) — same chain;
 *   3. the registry restore's descriptor wait (connections.ts) — resolves
 *      only when boot() publishes, so it is a parallel view of site 1.
 *
 * Sized by main's cold-start chain (desktop-boot-budget.ts): port announce,
 * health poll, and a margin for the work around them, so the IPC is still
 * in flight when a slow but healthy backend publishes. The accepted cost: a
 * wedged main-process round-trip at boot also takes this long to surface the
 * recovery overlay — the same trade main made raising the readiness poll to
 * 180s (#63454). The bound stays finite, so it ends.
 *
 * Main answers the announce deadline it resolved
 * (HERMES_DESKTOP_PORT_ANNOUNCE_TIMEOUT_MS included) through preload before
 * this module loads, so an override stretches both sides. The
 * undefined-window case (SSR, tests without a bridge) falls back to the
 * default deadline; the packaged app always reads the value main resolved.
 * Awaits against an already-spawned backend are reconnect-class and use
 * the shorter RECONNECT_ATTEMPT_TIMEOUT_MS below instead. */
export const BACKEND_BOOT_WAIT_TIMEOUT_MS = resolveRendererBootWaitMs(
  typeof window === 'undefined' ? undefined : window.hermesDesktop?.portAnnounceTimeoutMs
)

// desktop.getConnection() / getConnectionFor() / revalidateConnection() /
// resolveGatewayWsUrl() are IPC round-trips into the main process with no
// timeout of their own (#93454). A wedged main-process round-trip (e.g. a
// stuck revalidation after a liveness-probe trip) otherwise hangs an awaiting
// caller forever. Reconnect-class callers bound them with this budget;
// boot-class callers (see BACKEND_BOOT_WAIT_TIMEOUT_MS above) with the wider
// cold-spawn budget.
export const RECONNECT_ATTEMPT_TIMEOUT_MS = 20_000

/** Budget for the phase-1 dial of a source switch (store/connections
 * selectConnection). Unlike the two above this await is not one IPC: it is the
 * main process's whole remote bring-up chain — one ssh connect
 * (DEFAULT_CONNECT_TIMEOUT_MS 15 s in electron/ssh-connection.ts), the
 * sequential ssh execs that probe platform, locate the runtime and read its
 * version (DEFAULT_EXEC_TIMEOUT_MS 20 s each), the spawned backend's ready
 * sentinel (DEFAULT_READY_TIMEOUT_MS 45 s in electron/remote-lifecycle.ts) and
 * the port forward (DEFAULT_FORWARD_TIMEOUT_MS 15 s). Their sum is ~135 s, and
 * a *healthy* cold dial lands far inside it — measured 37-64 s against a 2 GB
 * VPS over Windows ssh (no mux: every stage is a fresh ssh process plus a cold
 * `serve` boot).
 *
 * Bounding that chain with RECONNECT_ATTEMPT_TIMEOUT_MS reported
 * "Could not connect to <source>" for switches that were still dialing, and
 * then connected anyway (withTimeout does not cancel the dial), which read as a
 * failure the user had to retry. The per-IPC budget still catches the #93454
 * class from inside the dial; this one only has to outlast a legitimate dial. */
export const SOURCE_SWITCH_DIAL_TIMEOUT_MS = 135_000

/** Rejection raised by withTimeout. The bounded work is NOT cancelled — the
 * caller decides what a straggler that settles later means. */
export class TimeoutError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'TimeoutError'
  }
}

export function isTimeoutError(error: unknown): error is TimeoutError {
  return error instanceof TimeoutError
}

/** Settle with `promise`, or reject with a TimeoutError after `ms`.
 * `onTimeout` runs synchronously before the rejection is published so callers
 * can revoke ownership of work that would otherwise keep running unowned. If
 * that callback throws, its error becomes this promise's rejection. */
export function withTimeout<T>(
  promise: Promise<T>,
  ms: number,
  message: string,
  onTimeout?: (error: TimeoutError) => void
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => {
      const error = new TimeoutError(message)

      try {
        onTimeout?.(error)
      } catch (onTimeoutError) {
        reject(onTimeoutError)

        return
      }

      reject(error)
    }, ms)

    Promise.resolve(promise).then(
      value => {
        clearTimeout(timer)
        resolve(value)
      },
      err => {
        clearTimeout(timer)
        reject(err)
      }
    )
  })
}

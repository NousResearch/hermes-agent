/**
 * first-run-gate.ts
 *
 * Pure, electron-free state machine for the desktop's first-run choice: on a
 * totally fresh machine (no install anywhere), the PRIMARY boot's ensureRuntime()
 * 'bootstrap-needed' branch parks here to let the user pick "install on this
 * computer" vs "connect to an existing server" BEFORE any local install starts.
 * Only the primary boot enables the gate (ensureRuntime's `firstRunGate` option);
 * pooled/secondary profiles bootstrap directly since they render no overlay.
 *
 * Kept standalone (no `import 'electron'`) so it can be unit-tested with
 * `node --test` — same pattern as connection-config.ts. main.ts owns one gate
 * instance, wires `onChanged` to the `hermes:first-run:changed` broadcast, and
 * bridges the decision into the boot flow:
 *   - chooseInstall() (IPC hermes:first-run:choose 'install') → proceed with the
 *     existing bootstrap unchanged.
 *   - abort() (IPC hermes:connection-config:apply, remote branch) → the pending
 *     ensureRuntime run throws a non-latching firstRunAborted error and
 *     transparently re-dials into the freshly-saved remote connection.
 *
 * The gate is a one-shot latch that RESETS after each decision, so a later
 * bootstrap-needed in the same process (e.g. a retry after a reload) asks again.
 */

export type FirstRunDecision = 'install' | 'abort'

/**
 * Create a first-run gate.
 *
 * @param {(state: { required: boolean }) => void} [onChanged] Notified with the
 *   current `{ required }` state on every transition (entering the wait, and on
 *   resolution). Fires exactly once per transition; never on a no-op call.
 * @returns a gate with `state()`, `waitForDecision()`, `chooseInstall()`, and
 *   `abort()`.
 */
function createFirstRunGate(onChanged?: (state: { required: boolean }) => void) {
  // The single in-flight decision, or null when not waiting. Holding the
  // promise here makes waitForDecision() idempotent (concurrent boot callers
  // share one wait) and lets chooseInstall()/abort() resolve it out of band.
  let pending: { promise: Promise<FirstRunDecision>; resolve: (decision: FirstRunDecision) => void } | null = null

  function state() {
    return { required: pending !== null }
  }

  function emitChanged() {
    if (typeof onChanged === 'function') {
      onChanged(state())
    }
  }

  // Enter (or re-join) the waiting state. Idempotent: repeat calls return the
  // same promise and do NOT re-fire onChanged, so the overlay isn't flickered.
  function waitForDecision(): Promise<FirstRunDecision> {
    if (pending) {
      return pending.promise
    }

    let resolveFn: (decision: FirstRunDecision) => void = () => {}

    const promise = new Promise<FirstRunDecision>(resolve => {
      resolveFn = resolve
    })

    pending = { promise, resolve: resolveFn }
    // Fire {required:true} only once, on the transition into the wait.
    emitChanged()

    return promise
  }

  // Resolve the wait and reset to the idle state so a later bootstrap-needed
  // asks again. No-op (no onChanged, no throw) when not waiting.
  function settle(decision: FirstRunDecision) {
    if (!pending) {
      return
    }

    const { resolve } = pending
    pending = null
    // Fire {required:false} before resolving so a listener that reacts to the
    // broadcast sees the idle state.
    emitChanged()
    resolve(decision)
  }

  function chooseInstall() {
    settle('install')
  }

  function abort() {
    settle('abort')
  }

  return { abort, chooseInstall, state, waitForDecision }
}

export { createFirstRunGate }

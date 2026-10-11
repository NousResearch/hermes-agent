import { GatewayReauthRequiredError, JsonRpcGatewayError } from '@hermes/shared'

import { isTimeoutError } from '@/lib/with-timeout'
import { $gatewaySwitching } from '@/store/gateway-switch'
import { $gatewayState } from '@/store/session'

/**
 * The welcome chat's start, as explicit states. While the kickoff is in a state it knows how to wait on
 * (the backend process is alive, the gateway is open or reconnecting, a call is in flight) it waits with no
 * wall-clock deadline: a cold first boot answers late, but it answers (51 s to listen on a Windows ARM64
 * first boot, then a busy backend behind the first calls). It fails only on what waiting cannot fix: an
 * RPC error response, the backend process exiting, a result it does not expect, or `stop` (the person chose
 * to go on without setup once the wait outlasted the boot budget).
 */
type KickoffState = 'waiting-backend' | 'preparing' | 'opening' | 'started' | 'off' | 'failed'

/** json-rpc-channel arms a call's timer only for a positive timeout, so 0 waits for the answer. */
export const NO_DEADLINE = 0

/** A failure no amount of waiting fixes. */
export class KickoffFailure extends Error {}

/** The person chose to go on without setup while the start was still waiting. */
export class KickoffSkipped extends KickoffFailure {}

type FailureKind = 'fatal' | 'starting' | 'transient'

function failureKind(error: unknown): FailureKind {
  if (
    error instanceof KickoffFailure ||
    error instanceof JsonRpcGatewayError ||
    error instanceof GatewayReauthRequiredError
  ) {
    return 'fatal'
  }

  // The renderer's own dial budget ran out while main still spawns the profile's backend. The spawn
  // goes on and answers with a connection or an error, so dialing again is waiting, not retrying.
  if (isTimeoutError(error)) {
    return 'starting'
  }

  // A socket that closed or reset under the call.
  return 'transient'
}

const describe = (error: unknown) => (error instanceof Error ? error.message : String(error))

function log(line: string): void {
  const text = `[setup] kickoff ${line}`
  console.info(text)
  // Only error-level console lines reach desktop.log; testers read the transitions there.
  window.hermesDesktop?.logLine?.(text)
}

interface KickoffMachine {
  enter(next: KickoffState, detail?: string): void
  /** Runs `step` once the gateway is open. A dial that times out on a starting backend runs it again; a
   *  dropped connection runs it again once. Anything else, or the backend exiting, rejects. */
  attempt<T>(state: KickoffState, step: () => Promise<T>): Promise<T>
  dispose(): void
  /** Aborts every call that carries `signal` and fails the kickoff with `reason`. */
  stop(reason: KickoffFailure): void
  /** Passed to every kickoff RPC, so `stop` rejects the ones still in flight and any started after it. */
  signal: AbortSignal
}

export function createKickoffMachine(): KickoffMachine {
  let state: KickoffState | 'begin' = 'begin'
  let exitFailure: KickoffFailure | null = null
  const exitWaiters = new Set<(error: KickoffFailure) => void>()
  const stopper = new AbortController()
  let stopReason: KickoffFailure | null = null

  const stop = (reason: KickoffFailure) => {
    stopReason = reason
    stopper.abort(reason)

    for (const reject of exitWaiters) {
      reject(reason)
    }
  }

  const enter = (next: KickoffState, detail?: string) => {
    log(`${state} -> ${next}${detail ? ` (${detail})` : ''}`)
    state = next
  }

  const offExit = window.hermesDesktop?.onBackendExit(({ code, signal }) => {
    // A gateway switch stops the old backend on purpose.
    if ($gatewaySwitching.get()) {
      return
    }

    exitFailure = new KickoffFailure(
      `The Hermes backend stopped before the welcome chat opened (code ${code ?? 'none'}, signal ${signal ?? 'none'}).`
    )

    for (const reject of exitWaiters) {
      reject(exitFailure)
    }
  })

  const gatewayOpen = (): Promise<void> => {
    if (stopReason) {
      return Promise.reject(stopReason)
    }

    if (exitFailure) {
      return Promise.reject(exitFailure)
    }

    if ($gatewayState.get() === 'open') {
      return Promise.resolve()
    }

    enter('waiting-backend', `gateway ${$gatewayState.get()}`)

    return new Promise((resolve, reject) => {
      const stopState = $gatewayState.listen(next => {
        if (next === 'open') {
          settle()
          resolve()
        }
      })

      const fail = (error: KickoffFailure) => {
        settle()
        reject(error)
      }

      function settle() {
        stopState()
        exitWaiters.delete(fail)
      }

      exitWaiters.add(fail)
    })
  }

  const attempt = async <T>(next: KickoffState, step: () => Promise<T>): Promise<T> => {
    let retried = false
    let detail: string | undefined

    for (;;) {
      await gatewayOpen()
      enter(next, detail)

      try {
        return await step()
      } catch (error) {
        const kind = failureKind(error)

        if (stopReason) {
          throw stopReason
        }

        if (exitFailure) {
          throw exitFailure
        }

        if (kind === 'fatal' || (kind === 'transient' && retried)) {
          throw error
        }

        retried ||= kind === 'transient'
        detail = `${kind === 'starting' ? 'backend still starting' : 'retry'}: ${describe(error)}`
      }
    }
  }

  return { attempt, dispose: () => offExit?.(), enter, signal: stopper.signal, stop }
}

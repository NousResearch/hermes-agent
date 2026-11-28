/**
 * ssh-config-probe.ts
 *
 * Runs the local `ssh -G` config probe and decides whether a failure is
 * terminal (latched as SSH_CLIENT_FAILED, #103288) or transient.
 *
 * A probe killed by its own timeout is not evidence of a broken ssh client.
 * `ssh -G` only parses local config and normally answers in milliseconds, but
 * when the Mac sleeps mid-boot the probe's timer expires during sleep and
 * fires the moment the main process wakes, killing a healthy ssh. Latching
 * that as terminal left the desktop offline until a manual restart (30 h in
 * one observed case). So a timeout is retried once in place, then surfaced as
 * an ordinary retryable remote failure.
 *
 * Only a run of consecutive timeouts that each took about as long as the
 * timeout itself (a genuinely hanging ssh, e.g. a `Match exec` in
 * ssh_config) latches. Wall-clock time includes sleep, so a probe killed by
 * sleep shows an elapsed time far above its timeout and never counts: several
 * dark wakes in one night must not add up to a latch. Any success, and every
 * system resume, resets the run.
 */

import { sshClientFailedError } from './backend-start-failure'

/** Consecutive genuine (awake) probe timeouts that count as a hanging ssh client. */
export const SSH_CONFIG_PROBE_TIMEOUT_LATCH = 3

/**
 * A timed-out attempt whose wall-clock time exceeds this multiple of the
 * timeout spanned a sleep, so it says nothing about the ssh client.
 */
export const SSH_CONFIG_PROBE_SLEEP_FACTOR = 2

export const SSH_CONFIG_PROBE_TIMEOUT = 'ssh-config-probe-timeout'

/**
 * True when the probe was killed (execFile timeout sets `killed` and
 * `signal`) or execText reported its own timeout (`timedOut`). A missing
 * binary (ENOENT/EACCES) or a non-zero exit is a real client failure.
 */
export function isSshConfigProbeTimeout(error: unknown): boolean {
  if (!error || typeof error !== 'object') {
    return false
  }

  const e = error as { killed?: boolean; signal?: string | null; timedOut?: boolean }

  return e.timedOut === true || e.killed === true || (typeof e.signal === 'string' && e.signal.length > 0)
}

export function isSshConfigProbeTimeoutError(error: unknown): boolean {
  return (error as { kind?: string } | null | undefined)?.kind === SSH_CONFIG_PROBE_TIMEOUT
}

function sshConfigProbeTimeoutError(sshBinary: string, cause: unknown): Error {
  const detail = cause instanceof Error ? cause.message : String(cause ?? '')

  const error = new Error(
    `The local SSH client (${sshBinary}) did not finish resolving the connection config in time (${detail}); retrying.`
  ) as Error & { kind: string; retryable: boolean }

  error.kind = SSH_CONFIG_PROBE_TIMEOUT
  error.retryable = true

  return error
}

export interface SshConfigProbeOptions {
  timeoutLatch?: number
  now?: () => number
}

export class SshConfigProbe {
  private consecutiveTimeouts = 0
  private readonly timeoutLatch: number
  private readonly now: () => number

  constructor({ timeoutLatch = SSH_CONFIG_PROBE_TIMEOUT_LATCH, now = Date.now }: SshConfigProbeOptions = {}) {
    this.timeoutLatch = timeoutLatch
    this.now = now
  }

  /** Consecutive genuine timeouts counted toward the latch. */
  get timeouts(): number {
    return this.consecutiveTimeouts
  }

  /** Forget earlier timeouts, e.g. on system resume. */
  reset(): void {
    this.consecutiveTimeouts = 0
  }

  /**
   * Run `exec` (the `ssh -G` call, which must enforce `timeoutMs` itself).
   * Resolves with its output; rejects with a terminal sshClientFailedError or
   * a retryable probe-timeout error.
   */
  async run(sshBinary: string, timeoutMs: number, exec: () => Promise<string>): Promise<string> {
    let lastTimeout: unknown

    // One immediate in-place retry: a wake-from-sleep timeout then costs one
    // more probe instead of a full outer boot cycle.
    for (let attempt = 0; attempt < 2; attempt++) {
      const startedAt = this.now()

      try {
        const output = await exec()
        this.consecutiveTimeouts = 0

        return output
      } catch (error) {
        if (!isSshConfigProbeTimeout(error)) {
          throw sshClientFailedError(sshBinary, error)
        }

        lastTimeout = error

        const elapsed = this.now() - startedAt

        if (elapsed < timeoutMs * SSH_CONFIG_PROBE_SLEEP_FACTOR) {
          this.consecutiveTimeouts += 1

          if (this.consecutiveTimeouts >= this.timeoutLatch) {
            throw sshClientFailedError(sshBinary, error)
          }
        }
      }
    }

    throw sshConfigProbeTimeoutError(sshBinary, lastTimeout)
  }
}

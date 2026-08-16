import type { ScanOutcome } from './venv-blocker-scan'

const DRAIN_SETTLE_MS = 500
const MAX_DRAIN_PASSES = 3

export interface WindowsUpdateForceDrainDeps {
  forceKillProcessTree: (pid: number) => void
  scan: () => Promise<ScanOutcome>
  wait?: (delayMs: number) => Promise<void>
}

function wait(delayMs: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, delayMs))
}

function blockerPids(outcome: Extract<ScanOutcome, { kind: 'blocked' }>): number[] {
  return [
    ...new Set(
      outcome.result.processes
        .filter(process => process.forceDrainEligible)
        .map(process => process.pid)
        .filter((pid): pid is number => Number.isInteger(pid) && pid > 0)
    )
  ]
}

/**
 * Force-drain the Hermes processes proven by the existing target-venv scan.
 *
 * The scan owns process eligibility; this helper only applies the existing
 * Windows tree killer and rechecks the same scanner a bounded number of times.
 */
export async function forceDrainWindowsUpdateBlockers(
  initial: ScanOutcome,
  { forceKillProcessTree, scan, wait: sleep = wait }: WindowsUpdateForceDrainDeps
): Promise<ScanOutcome> {
  let outcome = initial

  for (let pass = 0; pass < MAX_DRAIN_PASSES && outcome.kind === 'blocked'; pass += 1) {
    const pids = blockerPids(outcome)

    if (pids.length === 0) {
      break
    }

    for (const pid of pids) {
      try {
        forceKillProcessTree(pid)
      } catch {
        // The next scan is authoritative: a missing or protected process must
        // leave the update blocked instead of guessing at another target.
      }
    }

    await sleep(DRAIN_SETTLE_MS)
    outcome = await scan()
  }

  return outcome
}

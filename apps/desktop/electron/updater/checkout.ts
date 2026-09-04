// updater/checkout.ts — the git-checkout strategies (win + posix + manual).
//
// These wrap the existing bodies in main.ts rather than re-implementing
// them: the checkout update flow is deeply entangled with the god-file's
// closure helpers (resolveHealedBranch, venv-lock release, blocker scans,
// hand-off marker claims). The strategy adapter delegates; the ladder's
// TAIL (manual `hermes update` command card) lives here because it is pure
// enough to test.

import type { UpdaterApplyResultWire, UpdaterStatusWire } from './index'

export interface CheckoutStrategyDeps {
  /** The existing checkUpdates git-checkout body (unchanged). */
  checkBody: () => Promise<UpdaterStatusWire>
  /** The existing applyUpdates git-checkout body (unchanged). */
  applyBody: (opts: { stopSafeBlockers?: boolean }) => Promise<UpdaterApplyResultWire>
  isWindows: boolean
}

export class CheckoutStrategy {
  readonly mechanism: 'windows-handoff' | 'posix-handoff'

  constructor(private readonly deps: CheckoutStrategyDeps) {
    this.mechanism = deps.isWindows ? 'windows-handoff' : 'posix-handoff'
  }

  async check(): Promise<UpdaterStatusWire> {
    const status = await this.deps.checkBody()
    status.mechanism = this.mechanism

    return status
  }

  async apply(opts: { stopSafeBlockers?: boolean }): Promise<UpdaterApplyResultWire> {
    const result = await this.deps.applyBody(opts)
    result.mechanism = this.mechanism

    return result
  }
}

/**
 * The manual command card for a checkout with no staged updater: the exact
 * `hermes update` line to run, branch-pinned to the checkout's current branch
 * for non-main (bare `hermes update` would silently switch the install
 * off-branch). Extracted from applyUpdates so the wording contract is
 * unit-testable.
 */
export function buildManualUpdateCommand(currentBranch: string | null | undefined): string {
  return currentBranch && currentBranch !== 'HEAD' && currentBranch !== 'main'
    ? `hermes update --branch ${currentBranch}`
    : 'hermes update'
}

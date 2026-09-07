/**
 * Pure decision helper for the branch self-heal used by the desktop updater.
 *
 * `git ls-remote --exit-code --heads <remote> <branch>` exits 2 both when a
 * branch was pushed and later deleted (typically after being merged) AND when
 * the branch was never pushed at all — a purely local branch. Re-pinning the
 * updater to main is right in the first case and wrong in the second: for a
 * local-only branch the branch ref is the only place the user's commits exist,
 * so silently switching the running code to main discards their checkout.
 *
 * The primary fact is commitsBeyondMain: `git rev-list --count
 * origin/main..<branch>`, the number of commits the branch carries that main
 * lacks. Zero means every commit is already in main — healing to main loses
 * nothing, even when a pruned fetch already removed the remote-tracking ref
 * (the classic merged-and-deleted case). Any positive value means the branch
 * carries work main lacks, pushed or not — the pin must hold.
 *
 * When the commit count is unavailable (no origin/main ref, git error), the
 * remote-tracking ref decides: present means the branch was fetched/pushed
 * from the remote at least once, so "gone from the remote" reads as
 * merged-and-deleted; absent means git never saw the branch on the remote —
 * it is local-only and must keep its pin.
 *
 * Extracted from main.ts so the heal decision is unit testable without
 * booting Electron (main.ts requires('electron') at load).
 */

export type HealFacts = {
  /** Whether refs/remotes/origin/<branch> exists locally. */
  remoteTrackingRefExists: boolean
  /** `git rev-list --count origin/main..<branch>`; null when unavailable. */
  commitsBeyondMain: number | null
}

export type HealDecision = {
  /** Branch the updater should follow after the heal decision. */
  branch: string
  /** Short machine-readable reason for observability/logging. */
  reason:
    | 'healed-to-main' // nothing beyond main (or pushed-then-deleted fallback)
    | 'kept-local-branch' // carries commits main lacks, or never pushed
}

function decideHealedBranch(branch: string, facts: HealFacts): HealDecision {
  if (facts.commitsBeyondMain !== null) {
    return facts.commitsBeyondMain > 0
      ? { branch, reason: 'kept-local-branch' }
      : { branch: 'main', reason: 'healed-to-main' }
  }

  return facts.remoteTrackingRefExists
    ? { branch: 'main', reason: 'healed-to-main' }
    : { branch, reason: 'kept-local-branch' }
}

export { decideHealedBranch }

/** Pure policy for healing a configured desktop update branch that vanished upstream. */
export interface MissingUpdateBranchSignals {
  configuredBranch: string
  currentBranch: string
}

export function shouldHealMissingUpdateBranch(signals: MissingUpdateBranchSignals): boolean {
  const configured = signals.configuredBranch.trim()
  const current = signals.currentBranch.trim()

  if (!configured || configured === 'main') {
    return false
  }

  // A checkout currently on the missing branch may be a deliberately maintained
  // local upgrade lane. Silently switching its update target to main can launch a
  // destructive in-place merge and force the desktop to quit on failure.
  return !current || current !== configured
}

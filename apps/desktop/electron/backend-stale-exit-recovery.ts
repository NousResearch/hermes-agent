export type StaleExitRecoveryState = {
  hasCurrentProcess: boolean
  hasPendingStart: boolean
  intentionalTeardown: boolean
  recoveryClaimed: boolean
}

export function claimStaleBackendExitRecovery(state: StaleExitRecoveryState): boolean {
  if (state.hasCurrentProcess || state.hasPendingStart || state.intentionalTeardown || state.recoveryClaimed) return false
  state.recoveryClaimed = true
  return true
}

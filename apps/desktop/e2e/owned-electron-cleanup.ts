interface SandboxLaunchState {
  launchAttempted: boolean
}

export function cleanupAfterOwnedElectron(
  state: SandboxLaunchState,
  cleanup: () => void
): boolean {
  // Electron closure alone cannot establish that its backend has exited.
  if (state.launchAttempted) {
    return false
  }

  cleanup()

  return true
}

interface OwnedElectronCloseActions {
  close: () => Promise<void>
  killIfRunning: () => void
  waitForExit: () => Promise<void>
}

export async function finishOwnedElectronShutdown(actions: OwnedElectronCloseActions): Promise<void> {
  const failures: unknown[] = []

  try {
    await actions.close()
  } catch (error) {
    failures.push(error)

    try {
      actions.killIfRunning()
    } catch (killError) {
      failures.push(killError)
    }
  }

  try {
    await actions.waitForExit()
  } catch (error) {
    failures.push(error)
  }

  if (failures.length > 0) {
    throw new AggregateError(failures, 'Owned Electron close, fallback kill, or exit-poll failed')
  }
}

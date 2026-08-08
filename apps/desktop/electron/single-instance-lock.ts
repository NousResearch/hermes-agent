interface SingleInstanceApp {
  getPath(name: 'userData'): string
  requestSingleInstanceLock(): boolean
}

function singleInstanceLockDiagnostic(app: SingleInstanceApp): string {
  let userData = 'the Electron user-data directory'

  try {
    userData = app.getPath('userData')
  } catch {
    // The lock decision is authoritative even if its diagnostic path is unavailable.
  }

  return (
    '[hermes] Unable to acquire the Electron single-instance lock. ' +
    'Another Hermes Desktop instance may be running. If no instance is running, ' +
    `inspect SingletonLock, SingletonCookie, and SingletonSocket under ${userData}; ` +
    'Hermes will not remove these artifacts automatically because it cannot safely prove ownership.'
  )
}

export function requestSingleInstanceLockWithDiagnostic(app: SingleInstanceApp): boolean {
  const acquired = app.requestSingleInstanceLock()

  if (!acquired && process.platform === 'linux') {
    try {
      console.error(singleInstanceLockDiagnostic(app))
    } catch {
      // Logging must never change Electron's authoritative lock decision.
    }
  }

  return acquired
}

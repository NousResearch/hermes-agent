type BlockerKind = 'prevent-app-suspension'

export interface PowerSaveBlockerLike {
  start(type: BlockerKind): number
  isStarted(id: number): boolean
  stop(id: number): void
}

export interface PowerMonitorLike {
  on(event: 'lock-screen' | 'unlock-screen' | 'resume', listener: () => void): unknown
}

export interface PowerLifecycleGuard {
  readonly active: boolean
  start(): number | null
  stop(): boolean
}

export function createPowerLifecycleGuard(
  powerSaveBlocker: PowerSaveBlockerLike,
  log: (line: string) => void = () => undefined
): PowerLifecycleGuard {
  let blockerId: number | null = null

  return {
    get active() {
      return blockerId !== null && powerSaveBlocker.isStarted(blockerId)
    },
    start() {
      if (blockerId !== null && powerSaveBlocker.isStarted(blockerId)) {
        return blockerId
      }

      blockerId = powerSaveBlocker.start('prevent-app-suspension')
      log(`[power] app-suspension guard started id=${blockerId}`)

      return blockerId
    },
    stop() {
      if (blockerId === null) {
        return false
      }

      const id = blockerId

      if (powerSaveBlocker.isStarted(id)) {
        powerSaveBlocker.stop(id)
      }

      blockerId = null
      log(`[power] app-suspension guard stopped id=${id}`)

      return true
    }
  }
}

export function stopPowerLifecycleGuardSafely(
  guard: PowerLifecycleGuard,
  log: (line: string) => void = () => undefined
): boolean {
  try {
    return guard.stop()
  } catch (error) {
    try {
      log(`[power] app-suspension guard cleanup failed: ${String(error)}`)
    } catch {
      // Cleanup and its failure reporting must never interrupt lifecycle events.
    }

    return false
  }
}

export function registerPowerLifecycleListeners(
  powerMonitor: PowerMonitorLike,
  guard: PowerLifecycleGuard,
  onResume: () => void,
  log: (line: string) => void = () => undefined
): void {
  powerMonitor.on('resume', onResume)
  powerMonitor.on('lock-screen', () => {
    guard.start()
  })
  powerMonitor.on('unlock-screen', () => {
    stopPowerLifecycleGuardSafely(guard, log)
    onResume()
  })
}

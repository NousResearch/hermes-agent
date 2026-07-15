export type PreventSleepMode = 'display' | 'system'
export type PowerSaveBlockerType = 'prevent-app-suspension' | 'prevent-display-sleep'

export interface ResolvedPreventSleepConfig {
  enabled: boolean
  mode: PreventSleepMode
  surfaces: string[]
}

export interface PowerSaveBlockerLike {
  isStarted: (id: number) => boolean
  start: (type: PowerSaveBlockerType) => number
  stop: (id: number) => boolean
}

export interface PowerSaveControllerState extends ResolvedPreventSleepConfig {
  active: boolean
}

function normalizeSurface(value: unknown): string {
  return String(value ?? '')
    .trim()
    .toLowerCase()
    .replaceAll('_', '-')
}

function normalizeSurfaces(value: unknown): string[] {
  const items = Array.isArray(value) ? value : []

  const surfaces: string[] = []

  for (const item of items) {
    const surface = normalizeSurface(item)

    if (surface && !surfaces.includes(surface)) {
      surfaces.push(surface)
    }
  }

  return surfaces
}

export function resolvePreventSleepConfig(block: unknown, surface = 'desktop'): ResolvedPreventSleepConfig {
  const record = block && typeof block === 'object' ? (block as Record<string, unknown>) : {}
  const mode: PreventSleepMode = record.mode === 'display' ? 'display' : 'system'

  const surfaces = normalizeSurfaces(record.surfaces)

  return {
    enabled: record.enabled === true && surfaces.includes(normalizeSurface(surface)),
    mode,
    surfaces
  }
}

export function powerSaveTypeForMode(mode: PreventSleepMode): PowerSaveBlockerType {
  return mode === 'display' ? 'prevent-display-sleep' : 'prevent-app-suspension'
}

export function isPowerSaveRefreshAuthorized(sender: unknown, owner: unknown): boolean {
  return owner !== null && owner !== undefined && sender === owner
}

export function createPowerSaveBlockerController(powerSaveBlocker: PowerSaveBlockerLike, surface = 'desktop') {
  const blockers = new Map<number, PowerSaveBlockerType>()
  let config = resolvePreventSleepConfig(undefined, surface)

  const activeBlockers = (): [number, PowerSaveBlockerType][] => {
    for (const id of blockers.keys()) {
      if (!powerSaveBlocker.isStarted(id)) {
        blockers.delete(id)
      }
    }

    return [...blockers.entries()]
  }

  const isStarted = () => activeBlockers().length > 0

  const state = (): PowerSaveControllerState => {
    const active = activeBlockers()

    const mode = active.some(([, type]) => type === 'prevent-display-sleep')
      ? 'display'
      : active.length > 0
        ? 'system'
        : config.mode

    return { ...config, active: active.length > 0, mode }
  }

  const release = (id: number): boolean => {
    if (!blockers.has(id)) {
      return true
    }

    if (!powerSaveBlocker.isStarted(id)) {
      blockers.delete(id)

      return true
    }

    const stopped = powerSaveBlocker.stop(id)

    if (!stopped && powerSaveBlocker.isStarted(id)) {
      return false
    }

    if (powerSaveBlocker.isStarted(id)) {
      return false
    }

    blockers.delete(id)

    return true
  }

  const releaseAll = (ids: number[]): number[] => {
    const failed: number[] = []

    for (const id of ids) {
      try {
        if (!release(id)) {
          failed.push(id)
        }
      } catch {
        failed.push(id)
      }
    }

    return failed
  }

  const acquire = (type: PowerSaveBlockerType): number => {
    const id = powerSaveBlocker.start(type)

    if (!powerSaveBlocker.isStarted(id)) {
      throw new Error(`Electron did not start power save blocker ${id}`)
    }

    blockers.set(id, type)

    return id
  }

  const stop = (): boolean => {
    const ids = activeBlockers().map(([id]) => id)

    if (ids.length === 0) {
      return false
    }

    const failed = releaseAll(ids)

    if (failed.length > 0) {
      throw new Error(`Failed to stop ${failed.length} Electron power save blocker(s)`)
    }

    config = resolvePreventSleepConfig(undefined, surface)

    return true
  }

  const refresh = (block: unknown): PowerSaveControllerState & { changed: boolean } => {
    const next = resolvePreventSleepConfig(block, surface)
    const nextType = powerSaveTypeForMode(next.mode)
    const active = activeBlockers()
    const matching = active.filter(([, type]) => type === nextType)

    if (!next.enabled) {
      const failed = releaseAll(active.map(([id]) => id))

      if (failed.length > 0) {
        throw new Error(`Failed to stop ${failed.length} Electron power save blocker(s)`)
      }

      config = next

      return { ...state(), changed: active.length > 0 }
    }

    if (matching.length > 0) {
      const keepId = matching[0][0]
      const failed = releaseAll(active.filter(([id]) => id !== keepId).map(([id]) => id))

      if (failed.length > 0) {
        throw new Error(`Failed to replace ${failed.length} Electron power save blocker(s)`)
      }

      config = next

      return { ...state(), changed: active.length > 1 }
    }

    const nextId = acquire(nextType)
    const failed = releaseAll(active.map(([id]) => id))

    if (failed.length > 0) {
      const rollbackFailed = releaseAll([nextId])

      throw new Error(
        rollbackFailed.length > 0
          ? 'Failed to replace and roll back Electron power save blocker'
          : `Failed to replace ${failed.length} Electron power save blocker(s)`
      )
    }

    config = next

    return { ...state(), changed: true }
  }

  return { isStarted, refresh, state, stop }
}

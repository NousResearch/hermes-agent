import { type ChildProcess, spawn, type SpawnOptions } from 'node:child_process'

const DEFAULT_SPAWN_TIMEOUT_MS = 1500

interface WaitForUpdaterSpawnOptions {
  timeoutMs?: number
}

interface SpawnValidatedUpdaterOptions extends WaitForUpdaterSpawnOptions {
  spawnImpl?: (command: string, args: readonly string[], options: SpawnOptions) => ChildProcess
}

interface UpdaterChildEvents {
  off(event: 'error', listener: (error: Error) => void): unknown
  off(event: 'spawn', listener: () => void): unknown
  once(event: 'error', listener: (error: Error) => void): unknown
  once(event: 'spawn', listener: () => void): unknown
}

/**
 * Wait until Node confirms that a detached updater child was created. Spawn
 * failures are reported asynchronously on the child, so returning directly
 * from `spawn()` is not enough evidence to write the marker and quit Electron.
 */
export function waitForUpdaterSpawn(
  child: UpdaterChildEvents,
  { timeoutMs = DEFAULT_SPAWN_TIMEOUT_MS }: WaitForUpdaterSpawnOptions = {}
): Promise<void> {
  return new Promise((resolve, reject) => {
    let settled = false
    let timer: ReturnType<typeof setTimeout> | undefined

    const cleanup = () => {
      child.off('spawn', onSpawn)
      child.off('error', onError)

      if (timer) {
        clearTimeout(timer)
      }
    }

    const settle = <T>(callback: (value: T) => void) => (value: T) => {
      if (settled) {
        return
      }

      settled = true
      cleanup()
      callback(value)
    }

    const onSpawn = settle<void>(() => resolve())
    const onError = settle<Error>(error => reject(error))

    child.once('spawn', onSpawn)
    child.once('error', onError)
    timer = setTimeout(onSpawn, timeoutMs)
  })
}

/**
 * Spawn and validate a Windows updater handoff before detaching it. Both the
 * explicit update flow and bootstrap recovery use this boundary so neither
 * can write a live marker or quit after an asynchronous spawn failure.
 */
export async function spawnValidatedWindowsUpdater(
  command: string,
  args: readonly string[],
  spawnOptions: SpawnOptions,
  { spawnImpl = spawn, timeoutMs = DEFAULT_SPAWN_TIMEOUT_MS }: SpawnValidatedUpdaterOptions = {}
): Promise<ChildProcess> {
  const child = spawnImpl(command, args, spawnOptions)

  await waitForUpdaterSpawn(child, { timeoutMs })
  child.unref()

  return child
}

/**
 * Lifecycle coordination for lazily spawned profile backends.
 *
 * Kept separate from main.ts so the ownership races can be exercised with
 * Vitest without booting Electron. The Map entry is the pool's ownership
 * token: a child that exits late may only remove the exact entry it created.
 */

interface PoolChild {
  once(event: 'error', listener: (error: Error) => void): unknown
  once(event: 'exit', listener: (code: number | null, signal: string | null) => void): unknown
}

interface PoolEntry {
  process: unknown | null
}

interface PoolChildCallbacks {
  onError(error: Error): void
  onExit(code: number | null, signal: string | null): void
}

interface PoolTermination<Entry extends PoolEntry> {
  stopBackendChild(child: Entry['process']): void
  waitForBackendExit(child: Entry['process']): Promise<void>
}

export function deletePoolEntryIfOwned<Entry>(pool: Map<string, Entry>, profile: string, entry: Entry) {
  if (pool.get(profile) !== entry) {
    return false
  }

  return pool.delete(profile)
}

export function watchPoolBackendChild<Entry>(
  pool: Map<string, Entry>,
  profile: string,
  entry: Entry,
  child: PoolChild,
  callbacks: PoolChildCallbacks
) {
  child.once('error', error => {
    deletePoolEntryIfOwned(pool, profile, entry)
    callbacks.onError(error)
  })
  child.once('exit', (code, signal) => {
    deletePoolEntryIfOwned(pool, profile, entry)
    callbacks.onExit(code, signal)
  })
}

export async function terminatePoolBackendEntry<Entry extends PoolEntry>(
  entry: Entry,
  { stopBackendChild, waitForBackendExit }: PoolTermination<Entry>
) {
  stopBackendChild(entry.process)
  await waitForBackendExit(entry.process)
}

export async function cleanupFailedPoolBackend<Entry extends PoolEntry>(
  pool: Map<string, Entry>,
  profile: string,
  entry: Entry,
  termination: PoolTermination<Entry>
) {
  deletePoolEntryIfOwned(pool, profile, entry)
  await terminatePoolBackendEntry(entry, termination)
}

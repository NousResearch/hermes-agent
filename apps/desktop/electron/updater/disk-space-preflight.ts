import fs from 'node:fs/promises'

const MiB = 1024 * 1024
const GiB = 1024 * MiB

/** Headroom for the staged unpack, venv rebuild and the download itself. */
export const UPDATE_SPACE_NEEDED_BYTES = 2 * GiB

export interface DiskSpacePreflight {
  root: string
  log: (message: string) => void
  /** Injectable for tests. Defaults to statfs available bytes for an unprivileged user. */
  freeBytes?: (root: string) => Promise<number>
  /** Injectable for tests and configuration. Defaults to UPDATE_SPACE_NEEDED_BYTES. */
  neededBytes?: number
}

export async function freeBytesOn(root: string): Promise<number> {
  const stats = await fs.statfs(root)

  return stats.bavail * stats.bsize
}

function gib(bytes: number): string {
  return (bytes / GiB).toFixed(1)
}

/**
 * Fail-closed guard before an update mutates anything. Called while the
 * backend is still alive, next to preflightStateDb, so a full disk cannot
 * leave a half-migrated install. The error text names the cause, the exact
 * numbers and one concrete action, following bootstrap-failure-copy.ts.
 */
export async function preflightDiskSpace({
  root,
  log,
  freeBytes = freeBytesOn,
  neededBytes = UPDATE_SPACE_NEEDED_BYTES
}: DiskSpacePreflight): Promise<void> {
  try {
    const free = await freeBytes(root)

    if (free >= neededBytes) {
      log(`[updates] disk space pre-flight: ${gib(free)} GiB available in ${root}`)
      return
    }

    const message =
      `Brakuje miejsca na dysku. Aktualizacja potrzebuje około ${gib(neededBytes)} GiB ` +
      `wolnej przestrzeni w ${root}, a dostępne jest ${gib(free)} GiB. ` +
      'Twoja instalacja nie została zmieniona. Zwolnij miejsce, usuwając stare wersje i pliki tymczasowe, ' +
      'a potem wybierz Spróbuj ponownie.'

    log(`[updates] ${message}`)
    throw new Error(message)
  } catch (error: unknown) {
    if (error instanceof Error && error.message.startsWith('Brakuje miejsca')) {
      throw error
    }

    const message =
      `Nie udało się sprawdzić wolnej przestrzeni w ${root}: ` +
      `${error instanceof Error ? error.message : String(error)}. ` +
      'Aktualizacja została zatrzymana, zanim cokolwiek zmieniono. Spróbuj ponownie.'

    log(`[updates] ${message}`)
    throw new Error(message, { cause: error })
  }
}

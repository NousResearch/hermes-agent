/**
 * Decide which self-update path Desktop should use.
 *
 * Windows keeps the staged Hermes-Setup handoff because the running desktop can
 * hold the venv shim/exe open. POSIX platforms do not have that mandatory file
 * lock, and using an old staged helper can leave the desktop bundle's embedded
 * build stamp stale even after the backend checkout updates. So macOS/Linux run
 * the in-app update path: `hermes update` + `hermes desktop --build-only` +
 * relaunch/swap.
 */
export type DesktopUpdateStrategy = 'manual' | 'posix-in-app' | 'staged-updater'

export function chooseDesktopUpdateStrategy({
  isWindows,
  updater
}: {
  isWindows: boolean
  updater: string | null | undefined
}): DesktopUpdateStrategy {
  if (!isWindows) {
    return 'posix-in-app'
  }

  return updater ? 'staged-updater' : 'manual'
}

export function buildPosixUpdateArgs(branch: unknown): string[] {
  const normalized = typeof branch === 'string' ? branch.trim() : ''
  return normalized ? ['update', '--yes', '--branch', normalized] : ['update', '--yes']
}

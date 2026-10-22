/**
 * Tells the `hermes desktop` launcher that the main window is on screen.
 *
 * On Linux an app-grid launch defers writing the app's own `.desktop` entry
 * until the window has mapped: unpatched gnome-shell (before GNOME MR !4428)
 * drops a STARTING ShellApp's last reference when its entry changes, and the
 * next idle GC takes down the whole Wayland session (#111906). The launcher
 * hands us the write end of a pipe in `HERMES_DESKTOP_READY_FD`; one byte means
 * "mapped, safe to heal". A launch without the variable is a no-op.
 */

import fs from 'node:fs'

export const LAUNCHER_READY_FD_ENV = 'HERMES_DESKTOP_READY_FD'

type ReadyFdIo = {
  writeSync: (fd: number, data: string) => number
  closeSync: (fd: number) => void
}

/**
 * Signal the launcher once. The fd is closed after the write and the variable
 * removed from `env`, so a second reveal (or a child inheriting the env) can
 * never write into whatever file later reuses that descriptor number.
 */
export function notifyLauncherWindowRevealed(env: NodeJS.ProcessEnv = process.env, io: ReadyFdIo = fs): boolean {
  const raw = env[LAUNCHER_READY_FD_ENV]

  if (raw === undefined) {
    return false
  }

  delete env[LAUNCHER_READY_FD_ENV]
  const fd = Number(raw)

  if (!Number.isInteger(fd) || fd < 0) {
    return false
  }

  try {
    io.writeSync(fd, 'r')
    io.closeSync(fd)

    return true
  } catch {
    return false
  }
}

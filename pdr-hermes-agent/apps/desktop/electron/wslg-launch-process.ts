import { spawn, type StdioOptions } from 'node:child_process'
import { closeSync, fstatSync } from 'node:fs'
import * as inspector from 'node:inspector'

import { LAUNCHER_READY_FD_ENV } from './linux-launcher-ready'

/** Spawn the app while leaving this process alive only as its supervisor. */
export function spawnWslgLaunch(args: string[]) {
  const env = { ...process.env }
  const raw = env[LAUNCHER_READY_FD_ENV]
  delete env[LAUNCHER_READY_FD_ENV]
  delete process.env[LAUNCHER_READY_FD_ENV]

  let readyFd: number | undefined

  // Do not reinterpret empty/hex/fractional values, inherit stdio as a ready
  // pipe, or allocate a sparse stdio array from an untrusted descriptor number.
  if (raw !== undefined && /^\d+$/.test(raw)) {
    const fd = Number(raw)

    if (Number.isSafeInteger(fd) && fd >= 3) {
      try {
        fstatSync(fd)
        readyFd = fd
      } catch {
        // A stale descriptor must not prevent the desktop from starting.
      }
    }
  }

  const stdio: StdioOptions = ['inherit', 'inherit', 'inherit']

  if (readyFd !== undefined) {
    stdio.push(readyFd)
    env[LAUNCHER_READY_FD_ENV] = '3'
  }

  try {
    // Keep the original inspector argv/env for the app, but release the port
    // synchronously first: the waiting supervisor must not own its debugger.
    if (inspector.url()) {
      inspector.close()
    }

    return spawn(process.execPath, args, { stdio, env })
  } finally {
    // spawn duplicates the fd synchronously. Keeping our copy would suppress
    // EOF at the outer launcher until the entire desktop exits (also on error).
    if (readyFd !== undefined) {
      closeSync(readyFd)
    }
  }
}

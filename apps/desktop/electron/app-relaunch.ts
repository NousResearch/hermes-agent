import type { RelaunchOptions } from 'electron'

import { LAUNCHER_READY_FD_ENV } from './linux-launcher-ready'

interface RelaunchHost {
  relaunch(options: RelaunchOptions): void
}

/**
 * Every self-relaunch (GPU or sandbox fallback, update swap) goes through here.
 * The launcher's ready pipe is not inherited by the relaunched process, so a
 * carried `HERMES_DESKTOP_READY_FD` names whatever file reuses that descriptor
 * number there, and the reveal byte would be written into it.
 */
export function relaunchApp(host: RelaunchHost, options: RelaunchOptions, env: NodeJS.ProcessEnv = process.env): void {
  delete env[LAUNCHER_READY_FD_ENV]
  host.relaunch(options)
}

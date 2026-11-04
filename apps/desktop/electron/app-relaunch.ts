import type { ChildProcess } from 'node:child_process'

import type { RelaunchOptions } from 'electron'

import { LAUNCHER_READY_FD_ENV } from './linux-launcher-ready'
import { restoreLaunchTmpdir } from './linux-sandbox-relaunch'
import { canRelaunch, relaunchTarget, spawnRelaunch } from './relaunch-target'

interface RelaunchHost {
  relaunch(options: RelaunchOptions): void
}

/**
 * Every self-relaunch site (GPU fallbacks, update swap) goes through here; the
 * Linux --no-sandbox early restart calls createAppRelauncher's relaunch
 * directly and drops the fd itself. Both start the new process through
 * createAppRelauncher.
 * The launcher's ready pipe is not inherited by the relaunched process, so a
 * carried `HERMES_DESKTOP_READY_FD` names whatever file reuses that descriptor
 * number there, and the reveal byte would be written into it.
 */
export function relaunchApp(host: RelaunchHost, options: RelaunchOptions, env: NodeJS.ProcessEnv = process.env): void {
  delete env[LAUNCHER_READY_FD_ENV]
  host.relaunch(options)
}

export interface AppRelauncher {
  commandLine: { hasSwitch(name: string): boolean }
  relaunch(options: RelaunchOptions): void
  releaseSingleInstanceLock(): void
  exit(code: number): void
}

/**
 * The `app` every relaunch site and the final exit go through.
 *
 * On Linux, Electron's relauncher starts the new process with
 * PR_SET_NO_NEW_PRIVS, which every descendant inherits: the backend and the
 * agent's terminal can no longer use sudo, pkexec or any setuid helper, and a
 * FUSE-mounted AppImage cannot mount itself again. Linux therefore hands the
 * relaunch to an ordinary detached helper the moment it is requested, as
 * app.relaunch() does, so a fatal abort during the backend-shutdown wait still
 * comes back. The helper starts the target once this process has died, not
 * when Node's teardown closes its stdin (Chromium's shutdown, or a core dump,
 * can outlast that by seconds), so the new process neither races the old
 * backend, DevTools port or storage locks nor quits as a second instance. The
 * target is the AppImage file for a mounted image (the mount dies with this
 * process), otherwise this executable.
 */
export function createAppRelauncher(
  app: AppRelauncher,
  platform: NodeJS.Platform | string = process.platform,
  execPath: string = process.execPath,
  env: NodeJS.ProcessEnv = process.env
): AppRelauncher {
  if (platform !== 'linux') {
    return app
  }

  let helper: ChildProcess | null = null
  // JS keeps running for a moment after app.exit(); a relaunch requested then would start a second child.
  let exiting = false

  return {
    commandLine: app.commandLine,
    releaseSingleInstanceLock: () => app.releaseSingleInstanceLock(),
    relaunch: options => {
      if (exiting) {
        return
      }

      const args = options.args ?? process.argv.slice(1)
      const target = relaunchTarget(execPath, env)
      // The new process skips `hermes desktop`: it needs the launcher's short socket TMPDIR,
      // not the long one main.ts restored after the lock (linux-sandbox-relaunch.ts).
      const childEnv = { ...env }
      restoreLaunchTmpdir(childEnv)
      // The latest request wins. The cancel relies on the old helper's stdin staying open until the
      // SIGTERM lands: closing it first would let that helper exec, starting two targets.
      helper?.kill()
      helper = null

      if (canRelaunch(target, 'relaunching through Electron instead', childEnv)) {
        try {
          helper = spawnRelaunch(target, args, childEnv, execPath)

          return
        } catch (error) {
          // A synchronous spawn throw (E2BIG) must not cost the restart.
          console.error(
            `[hermes] relaunching ${target} failed (${String(error)}); relaunching through Electron instead`
          )
        }
      }

      app.relaunch({ args })
    },
    exit: code => {
      exiting = true

      // The helper's target takes the lock once this process is gone.
      if (helper) {
        app.releaseSingleInstanceLock()
      }

      app.exit(code)
    }
  }
}

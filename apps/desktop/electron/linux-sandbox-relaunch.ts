// A runtime `--no-sandbox` is too late on Linux (#131055, #139542, #143569).
//
// Chromium forks its zygote into the sandbox (user-namespace or setuid lane)
// before any application JavaScript runs, so `app.commandLine.appendSwitch(
// 'no-sandbox')` from main.ts reaches only the renderers' own command lines.
// Those renderers are still forked from the sandboxed zygote, but believe they
// are unsandboxed: they skip the sandbox's shared-memory path, open /dev/shm
// (or $TMPDIR under --disable-dev-shm-usage) directly from inside the
// zygote's empty chroot, get ESRCH, and die with "Try 'sudo chmod 1777
// /dev/shm'" — a crash loop with no window, or an unbounded log storm when
// the directory is not /dev/shm (#144638).
//
// The flag is honored only when it is on the browser's launch command line
// (or ELECTRON_DISABLE_SANDBOX is set at launch), so a Linux launch that asks
// for it at runtime — the boot-abort ladder's sticky fallback, or
// `desktop.electron_flags` read from config.yaml by a launch that bypassed
// `hermes desktop` — restarts once with the flag on argv instead.

import { LAUNCHER_READY_FD_ENV } from './linux-launcher-ready'
import { alreadyHasNoSandbox, buildNoSandboxRelaunchArgs } from './windows-sandbox-fallback'

export interface SandboxRelaunch {
  args: string[]
  execPath?: string
}

// Captured when this module is evaluated, before main.ts sets
// ELECTRON_DISABLE_SANDBOX for its children: only the launch-time state says
// whether Chromium actually started unsandboxed.
const LAUNCHED_WITHOUT_SANDBOX = alreadyHasNoSandbox(process.argv, process.env)

// `hermes desktop` shortens TMPDIR on Linux so the single-instance lock can
// bind its socket, parking the real one in HERMES_DESKTOP_TMPDIR; main.ts
// hands it back right after the lock. The relaunch does not go through the
// launcher again, so it must inherit the short launch pair, not the restored
// long TMPDIR that would hang its requestSingleInstanceLock().
const DESKTOP_TMPDIR_ENV = 'HERMES_DESKTOP_TMPDIR'

const LAUNCH_TMPDIR_PAIR = process.env[DESKTOP_TMPDIR_ENV]
  ? { socketTmpdir: process.env.TMPDIR, realTmpdir: process.env[DESKTOP_TMPDIR_ENV] }
  : null

export function restoreLaunchTmpdir(env: NodeJS.ProcessEnv, pair = LAUNCH_TMPDIR_PAIR): void {
  if (!pair) {
    return
  }

  env.TMPDIR = pair.socketTmpdir
  env[DESKTOP_TMPDIR_ENV] = pair.realTmpdir
}

export function linuxNoSandboxRelaunch(options: {
  platform: NodeJS.Platform | string
  runtimeNoSandbox: boolean
  argv: readonly string[]
  env: NodeJS.ProcessEnv
  launchedWithoutSandbox?: boolean
}): SandboxRelaunch | null {
  const launchedWithoutSandbox = options.launchedWithoutSandbox ?? LAUNCHED_WITHOUT_SANDBOX

  if (options.platform !== 'linux' || !options.runtimeNoSandbox || launchedWithoutSandbox) {
    return null
  }

  const args = buildNoSandboxRelaunchArgs(options.argv.slice(1))

  // An AppImage's execPath lives in a FUSE mount that disappears with this
  // process; relaunch the image itself.
  return options.env.APPIMAGE ? { args, execPath: options.env.APPIMAGE } : { args }
}

interface RelaunchHost {
  commandLine: { hasSwitch(name: string): boolean }
  relaunch(options: SandboxRelaunch): void
  exit(code: number): void
}

// Call once, in the lock-owning instance, after every runtime switch source
// (sandbox ladder, config.yaml) has run and before `ready`.
export function relaunchIfNoSandboxCameTooLate(host: RelaunchHost): boolean {
  const relaunch = linuxNoSandboxRelaunch({
    platform: process.platform,
    runtimeNoSandbox: host.commandLine.hasSwitch('no-sandbox'),
    argv: process.argv,
    env: process.env
  })

  if (!relaunch) {
    return false
  }

  console.warn('[hermes] --no-sandbox requested after Chromium started; relaunching with it on the command line')
  // The launcher's ready pipe is not inherited across the relaunch; a stale fd
  // number would make the next process write its reveal byte into whatever
  // file reuses that descriptor.
  delete process.env[LAUNCHER_READY_FD_ENV]
  restoreLaunchTmpdir(process.env)
  host.relaunch(relaunch)
  host.exit(0)

  return true
}

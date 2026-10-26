import { spawn } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

// AppImage is Linux-only; POSIX semantics keep the answer identical on every test host.
const posix = path.posix

function isInside(dir: string, file: string): boolean {
  const inside = posix.relative(dir, file)

  return Boolean(inside) && !inside.startsWith('..') && !posix.isAbsolute(inside)
}

function realpathOrNull(target: string): string | null {
  try {
    return fs.realpathSync(target)
  } catch {
    return null
  }
}

/**
 * The AppImage file this executable runs from, or undefined when it is not one.
 *
 * The AppImage runtime exports APPIMAGE/APPDIR to every descendant: Hermes
 * started from a terminal or IDE AppImage inherits that app's values, and
 * acting on them would relaunch, or rm -rf on uninstall, the other app.
 * APPIMAGE is ours only when this executable runs from that image's mount:
 * - inside APPDIR when it is set (the FUSE mount, or the
 *   `appimage_extracted_<hash>` dir of --appimage-extract-and-run);
 * - otherwise inside a type-2 runtime mount, `$TMPDIR/.mount_<name><random>`
 *   (AppImage/type2-runtime src/runtime/runtime.c build_mount_point():
 *   "%s/.mount_%.*sXXXXXX", then mkdtemp).
 */
export function ownAppImage(execPath: string, env: NodeJS.ProcessEnv): string | undefined {
  const { APPDIR: appDir, APPIMAGE: appImage } = env

  if (!appImage || !execPath) {
    return undefined
  }

  const exe = posix.resolve(execPath)

  if (!appDir) {
    return posix.dirname(exe).split('/').some(segment => segment.startsWith('.mount_')) ? appImage : undefined
  }

  if (isInside(posix.resolve(appDir), exe)) {
    return appImage
  }

  // The runtime builds APPDIR from the raw $TMPDIR, while process.execPath has
  // its symlinks resolved: TMPDIR=~/tmp on Fedora Atomic (/home -> /var/home).
  const realAppDir = realpathOrNull(appDir)
  const realExe = realpathOrNull(exe)

  return realAppDir && realExe && isInside(realAppDir, realExe) ? appImage : undefined
}

export type RelaunchStrategy =
  /** Electron's app.relaunch(): process.execPath again once this process exits. */
  | { kind: 'electron' }
  /** A FUSE-mounted AppImage: the image file again, started now as a detached child. */
  | { kind: 'appimage'; image: string }

/**
 * How this app starts itself again.
 *
 * Electron's relauncher execs its target with PR_SET_NO_NEW_PRIVS, so a
 * relaunched AppImage runtime cannot mount through the setuid fusermount
 * (EPERM, exit 127), and process.execPath lives in this process's FUSE mount,
 * which the runtime unmounts as soon as we exit. A mounted AppImage therefore
 * starts its image file as an ordinary child. Under --appimage-extract-and-run
 * APPDIR is a plain directory and the image would need the FUSE those users
 * lack, so the executable in it is relaunched as usual.
 */
export function relaunchStrategy(
  execPath: string = process.execPath,
  env: NodeJS.ProcessEnv = process.env
): RelaunchStrategy {
  const image = ownAppImage(execPath, env)

  if (!image || (env.APPDIR && posix.basename(posix.resolve(env.APPDIR)).startsWith('appimage_extracted_'))) {
    return { kind: 'electron' }
  }

  return { kind: 'appimage', image }
}

export interface RelaunchHost {
  relaunch(options: { args: string[] }): void
  releaseSingleInstanceLock(): void
}

/**
 * Whether `strategy` can start this app again. A spawn failure is reported
 * asynchronously, after the caller has already exited, so the image is
 * checked up front and a refusal is logged now; the caller then keeps running
 * in place of the relaunch.
 */
export function canRelaunch(strategy: RelaunchStrategy): boolean {
  if (strategy.kind === 'electron') {
    return true
  }

  try {
    fs.accessSync(strategy.image, fs.constants.X_OK)

    return true
  } catch (error) {
    const reason = (error as NodeJS.ErrnoException).code ?? String(error)
    console.error(`[hermes] cannot start ${strategy.image} again (${reason}); continuing without the relaunch`)

    return false
  }
}

// libuv's spawn keeps every descriptor this process inherited without
// O_CLOEXEC, which Chromium's launcher (app.relaunch) closes. Ours include the
// AppImage runtime's keepalive pipe, whose last reader keeps this process's
// FUSE mount and squashfuse daemon alive, and the launcher's ready pipe. The
// child therefore goes through /bin/sh, which drops every fd above 2 and then
// execs the image. POSIX only promises redirections on fds 0-9 and dash
// exits on `exec 10<&-`, so higher numbers are closed where the shell takes
// them (bash, busybox) and left in place under dash.
const EXEC_WITH_STDIO_ONLY = `big=
(eval 'exec 10<&-') 2>/dev/null && big=1
for f in /proc/self/fd/*; do
  n=\${f##*/}
  case $n in
    0 | 1 | 2) continue ;;
    [3-9]) ;;
    *) [ -n "$big" ] || continue ;;
  esac
  eval "exec $n<&-"
done
exec "$0" "$@"`

/**
 * `env` without the PATH-style entries inside this mount, which the image's
 * AppRun prepended (PATH, LD_LIBRARY_PATH, XDG_DATA_DIRS, ...). The new
 * image's AppRun prepends its own, while this mount goes away as soon as the
 * child stops holding it: a lookup through it while its FUSE daemon exits
 * fails with ENOTCONN, which aborted the new AppRun's `env bash`.
 */
function withoutMountEntries(env: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  const appDir = env.APPDIR && posix.resolve(env.APPDIR)

  if (!appDir) {
    return env
  }

  const inMount = (entry: string) => entry === appDir || entry.startsWith(`${appDir}/`)
  const cleaned: NodeJS.ProcessEnv = { ...env }

  for (const [key, value] of Object.entries(env)) {
    const entries = value?.split(':') ?? []

    if (!entries.some(inMount)) {
      continue
    }

    const kept = entries.filter(entry => !inMount(entry)).join(':')

    if (kept) {
      cleaned[key] = kept
    } else {
      delete cleaned[key]
    }
  }

  return cleaned
}

/**
 * Start this app again with `args`; the caller exits right after.
 * Check `canRelaunch(strategy)` first. The detached child outlives this
 * process (and its mount) and takes the single-instance lock itself, so the
 * lock is dropped first.
 */
export function relaunchNow(
  host: RelaunchHost,
  args: string[],
  strategy: RelaunchStrategy = relaunchStrategy(),
  env: NodeJS.ProcessEnv = process.env
): void {
  if (strategy.kind === 'electron') {
    host.relaunch({ args })

    return
  }

  host.releaseSingleInstanceLock()

  const child = spawn('/bin/sh', ['-c', EXEC_WITH_STDIO_ONLY, strategy.image, ...args], {
    detached: true,
    stdio: 'ignore',
    env: withoutMountEntries(env)
  })

  child.on('error', error => console.error(`[hermes] relaunching ${strategy.image} failed: ${error.message}`))
  child.unref()
}

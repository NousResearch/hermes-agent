import path from 'node:path'

// AppImage is Linux-only; POSIX semantics keep the answer identical on every test host.
const posix = path.posix

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

  const inside = posix.relative(posix.resolve(appDir), exe)

  return inside && !inside.startsWith('..') && !posix.isAbsolute(inside) ? appImage : undefined
}

/**
 * The executable that starts this app again, or undefined for Electron's
 * default (process.execPath). A real AppImage runs from a mount that
 * disappears when this process exits, so it relaunches the image file instead.
 */
export function relaunchExecPath(
  execPath: string = process.execPath,
  env: NodeJS.ProcessEnv = process.env
): string | undefined {
  return ownAppImage(execPath, env)
}

import path from 'node:path'

/**
 * The executable that starts this app again, or undefined for Electron's
 * default (process.execPath).
 *
 * A real AppImage runs from a FUSE mount (APPDIR) that disappears when this
 * process exits, so it relaunches the image file ($APPIMAGE) instead. The
 * AppImage runtime exports APPIMAGE/APPDIR to every descendant, though: Hermes
 * started from a terminal or IDE AppImage inherits that app's pair, and
 * relaunching it would start the other app while Hermes never comes back. The
 * pair is ours only when this executable lives inside APPDIR.
 */
export function relaunchExecPath(
  execPath: string = process.execPath,
  env: NodeJS.ProcessEnv = process.env
): string | undefined {
  const { APPDIR: appDir, APPIMAGE: appImage } = env

  if (!appImage || !appDir) {
    return undefined
  }

  const inside = path.relative(path.resolve(appDir), path.resolve(execPath))

  return inside && !inside.startsWith('..') && !path.isAbsolute(inside) ? appImage : undefined
}

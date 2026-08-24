import fs from 'node:fs'
import path from 'node:path'

import { nativeImage } from 'electron'

/**
 * Validate that a candidate app-icon file exists and decodes as an image.
 *
 * Electron's `new BrowserWindow({ icon })` and `app.dock.setIcon()` decode the
 * file synchronously on the main process and THROW when the bytes are not a
 * decodable image — `statSync().isFile()` only proves the file exists, not that
 * it decodes. A truncated or zero-byte PNG inside a packaged `app.asar` (e.g.
 * interrupted electron-builder run) therefore killed the main process inside
 * `createWindow()` and took the whole app down mid-session: the window never
 * appeared, running turns lost their renderer, and the desktop log showed
 * `Uncaught exception: Error: Failed to load image from path
 * '.../app.asar/public/apple-touch-icon.png' at createWindow`.
 *
 * This helper makes icon resolution fail-soft: a candidate that exists but does
 * not decode is skipped like a missing one, so the app falls through to the
 * next candidate (or starts with the platform default icon) instead of dying.
 * `nativeImage.createFromPath` is Electron's own decoder with the same failure
 * mode, so callers can inject a probe matching their environment; the shipped
 * probe decodes eagerly and treats a thrown error OR an empty image as invalid.
 */
export type IconProbe = (filePath: string) => boolean

/** Eager-decoding default probe: the file must decode to a non-empty image. */
export function decodingFileProbe(filePath: string): boolean {
  try {
    if (!fs.statSync(filePath).isFile()) {
      return false
    }
  } catch {
    return false
  }

  try {
    return !nativeImage.createFromPath(filePath).isEmpty()
  } catch {
    return false
  }
}

/**
 * Pick the first app-icon candidate that exists AND decodes; `undefined` when
 * none do (callers already treat a missing icon as optional — `if (icon)`).
 *
 * Pure over `(candidates, probe)` so the precedence ladder is unit-testable
 * without a running Electron app; the shipped probe injects the real decoder.
 */
export function resolveAppIcon(
  candidates: readonly string[],
  probe: IconProbe = decodingFileProbe
): string | undefined {
  for (const candidate of candidates) {
    if (probe(candidate)) {
      return candidate
    }
  }

  return undefined
}

/**
 * Build the platform-aware candidate ladder shared by every window factory.
 * Kept next to the resolver so precedence has one home; `appRoot` is injected
 * (packaged `APP_ROOT` vs dev tree) and `unpackedPathFor` maps into
 * `app.asar.unpacked` for builds that leave assets outside the archive.
 */
export function appIconCandidates(opts: {
  isWindows: boolean
  appRoot: string
  resourcesPath?: string
  unpackedPathFor: (p: string) => string
}): string[] {
  const { isWindows, appRoot, resourcesPath, unpackedPathFor } = opts

  return [
    ...(isWindows ? [path.join(resourcesPath ?? '', 'icon.ico'), path.join(appRoot, 'assets', 'icon.ico')] : []),
    path.join(appRoot, 'public', 'apple-touch-icon.png'),
    path.join(appRoot, 'dist', 'apple-touch-icon.png'),
    path.join(unpackedPathFor(appRoot), 'dist', 'apple-touch-icon.png')
  ]
}

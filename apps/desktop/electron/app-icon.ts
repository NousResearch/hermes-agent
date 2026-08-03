import path from 'node:path'

/**
 * Resolve the native BrowserWindow / dock icon path.
 *
 * Why this order matters on Windows: the taskbar / Start Menu / .lnk identity
 * comes from the PE icon stamped onto Hermes.exe (assets/icon.ico via
 * set-exe-identity / extraResources). BrowserWindow `{ icon }` must prefer that
 * same .ico (or resources/icon.ico) — not apple-touch-icon.png — or Alt+Tab /
 * window grouping can show a different glyph than the taskbar.
 *
 * On Darwin, `extraResources` still ships `resources/icon.ico` (the Windows
 * PE-stamp source) for every target. Prefer `.icns` then PNG ahead of that
 * `.ico` so the dock keeps a crisp native candidate when both exist.
 */

export type AppIconPathOptions = {
  appRoot: string
  resourcesPath?: string | null
  /** When APP_ROOT is inside app.asar, Electron still serves sibling files. */
  unpackedAppRoot?: string | null
  platform?: NodeJS.Platform
}

/**
 * Candidate absolute paths, highest priority first.
 * Pure: no fs I/O — caller picks the first existing path.
 *
 * Aligns with post-80c86c4 main.ts Windows `.ico`-first behaviour while making
 * Darwin priority platform-aware (`.icns` / PNG before packaged `.ico`).
 */
export function appIconCandidates({
  appRoot,
  resourcesPath = null,
  unpackedAppRoot = null,
  platform = process.platform
}: AppIconPathOptions): string[] {
  const roots = [appRoot, unpackedAppRoot].filter((r): r is string => Boolean(r))
  const out: string[] = []

  // 1) Packaged native format.
  if (resourcesPath) {
    if (platform === 'darwin') {
      out.push(path.join(resourcesPath, 'icon.icns'))
      out.push(path.join(resourcesPath, 'icon.png'))
    } else {
      out.push(path.join(resourcesPath, 'icon.ico'))
    }
  }

  // 2) Dev / asar-packaged assets next to the app (files: assets/**).
  for (const root of roots) {
    if (platform === 'darwin') {
      out.push(path.join(root, 'assets', 'icon.icns'))
      out.push(path.join(root, 'assets', 'icon.png'))
    } else {
      out.push(path.join(root, 'assets', 'icon.ico'))
      out.push(path.join(root, 'assets', 'icon.png'))
    }
  }

  // 3) Renderer favicon last — historically used, but can drift from the
  //    stamped .ico; keep as a soft fallback so windows still get *an* icon.
  for (const root of roots) {
    out.push(path.join(root, 'public', 'apple-touch-icon.png'))
    out.push(path.join(root, 'dist', 'apple-touch-icon.png'))
  }

  // Every Darwin-native or PNG candidate must stay ahead of the Windows PE
  // stamp source, even when the native icon lives outside resources/.
  if (platform === 'darwin') {
    if (resourcesPath) {
      out.push(path.join(resourcesPath, 'icon.ico'))
    }

    for (const root of roots) {
      out.push(path.join(root, 'assets', 'icon.ico'))
    }
  }

  // Dedupe while preserving order.
  const seen = new Set<string>()
  const unique: string[] = []

  for (const candidate of out) {
    const key = platform === 'win32' ? candidate.toLowerCase() : candidate

    if (seen.has(key)) {
      continue
    }

    seen.add(key)
    unique.push(candidate)
  }

  return unique
}

export function resolveAppIconPath(
  options: AppIconPathOptions,
  exists: (filePath: string) => boolean
): string | undefined {
  return appIconCandidates(options).find(exists)
}

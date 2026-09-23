import fs from 'node:fs'
import path from 'node:path'

import { resolveRemovableAppPath } from './desktop-uninstall'
import { isPackagedInstallPath as isPackagedInstallPathUnderRoots } from './workspace-cwd'

interface DesktopWorkspaceCwdRuntimeDeps {
  app: { getPath: (name: string) => string }
  APP_ROOT: string
  SOURCE_REPO_ROOT: string
  IS_PACKAGED: boolean
  directoryExists: (directoryPath: string) => boolean
  rememberLog: (message: string) => void
}

export function createDesktopWorkspaceCwdRuntime(deps: DesktopWorkspaceCwdRuntimeDeps) {
  const { app, APP_ROOT, SOURCE_REPO_ROOT, IS_PACKAGED, directoryExists, rememberLog } = deps

  function isPackagedInstallPath(dir) {
    return isPackagedInstallPathUnderRoots(dir, {
      isPackaged: IS_PACKAGED,
      installRoots: [
        APP_ROOT,
        path.dirname(process.execPath),
        resolveRemovableAppPath(process.execPath, process.platform, process.env)
      ]
    })
  }

  function resolveHermesCwd() {
    // In a packaged build, `process.cwd()` resolves to the install root (e.g.
    // `…/win-unpacked` on Windows or `/Applications/Hermes.app/Contents/...`
    // on macOS). Sessions spawned there leave files inside the app bundle
    // and bewilder users when "where did my files go?" is the install dir.
    // The user-configurable default project directory wins over everything,
    // followed by env hints (only honored when packaged if they point at a
    // real directory), then the home dir.
    const candidates = [
      readDefaultProjectDir(),
      process.env.HERMES_DESKTOP_CWD,
      IS_PACKAGED ? null : process.env.INIT_CWD,
      IS_PACKAGED ? null : process.cwd(),
      !IS_PACKAGED ? SOURCE_REPO_ROOT : null,
      app.getPath('home')
    ]

    for (const candidate of candidates) {
      if (!candidate) {
        continue
      }

      const resolved = path.resolve(String(candidate))

      if (isPackagedInstallPath(resolved)) {
        continue
      }

      if (directoryExists(resolved)) {
        return resolved
      }
    }

    return app.getPath('home')
  }

  function sanitizeWorkspaceCwd(cwd) {
    const trimmed = typeof cwd === 'string' ? cwd.trim() : ''

    if (!trimmed || isPackagedInstallPath(trimmed)) {
      return { cwd: resolveHermesCwd(), sanitized: Boolean(trimmed) }
    }

    try {
      const resolved = path.resolve(trimmed)

      if (directoryExists(resolved)) {
        return { cwd: resolved, sanitized: false }
      }
    } catch {
      // Fall through to the resolved default.
    }

    return { cwd: resolveHermesCwd(), sanitized: Boolean(trimmed) }
  }

  // Persisted "Default project directory" — surfaced as a setting in the
  // renderer (see app/settings/sessions-settings.tsx). Stored as JSON in
  // userData so it survives self-updates without bleeding into the new
  // install. `null` means "no preference, fall back to the usual chain".
  const DEFAULT_PROJECT_DIR_CONFIG_FILENAME = 'project-dir.json'

  function defaultProjectDirConfigPath() {
    return path.join(app.getPath('userData'), DEFAULT_PROJECT_DIR_CONFIG_FILENAME)
  }

  function readDefaultProjectDir() {
    try {
      const raw = fs.readFileSync(defaultProjectDirConfigPath(), 'utf8')
      const parsed = JSON.parse(raw)

      if (parsed && typeof parsed.dir === 'string' && parsed.dir.trim()) {
        const resolved = path.resolve(parsed.dir)

        if (directoryExists(resolved)) {
          return resolved
        }
      }
    } catch {
      // Missing / unreadable / malformed → fall through to the rest of the
      // candidate chain.
    }

    return null
  }

  function writeDefaultProjectDir(dir) {
    const target = defaultProjectDirConfigPath()
    const payload = dir ? JSON.stringify({ dir: path.resolve(dir) }, null, 2) : JSON.stringify({}, null, 2)

    try {
      fs.mkdirSync(path.dirname(target), { recursive: true })
      fs.writeFileSync(target, payload, 'utf8')
    } catch (error) {
      rememberLog(`[settings] write default project dir failed: ${error.message}`)
    }
  }

  return { isPackagedInstallPath, resolveHermesCwd, sanitizeWorkspaceCwd, readDefaultProjectDir, writeDefaultProjectDir }
}

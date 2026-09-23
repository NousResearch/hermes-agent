import fs from 'node:fs'
import path from 'node:path'

interface DesktopInstallHomeRuntimeDeps {
  APP_ROOT: string
  USER_DATA_OVERRIDE: string | undefined
  IS_WINDOWS: boolean
  app: { getPath: (name: string) => string }
  directoryExists: (directoryPath: string) => boolean
  normalizeHermesHomeRoot: (root: string) => string
  readWindowsUserEnvVar: (name: string) => string | null
}

export function createDesktopInstallHomeRuntime(deps: DesktopInstallHomeRuntimeDeps) {
  const {
    APP_ROOT,
    USER_DATA_OVERRIDE,
    IS_WINDOWS,
    app,
    directoryExists,
    normalizeHermesHomeRoot,
    readWindowsUserEnvVar
  } = deps

  // Build-time install stamp -- the git ref this .exe was built against.
  //
  // Written by apps/desktop/scripts/write-build-stamp.mjs during `npm run build`
  // and bundled into packaged apps via electron-builder's extraResources entry,
  // so the runtime stamp ends up at process.resourcesPath/install-stamp.json
  // after install. The bootstrap runner (Phase 1D) reads it to know which
  // commit to clone when running install.ps1 stages at first launch.
  //
  // Returns null when the file is missing (dev runs from a checkout where
  // build hasn't been invoked, or schema mismatch). Callers must handle null.
  //
  // Schema:
  //   { schemaVersion: 1, commit, branch, builtAt, dirty, source }
  const INSTALL_STAMP_SCHEMA_VERSION = 1

  function loadInstallStamp() {
    // Try packaged location first (resources/install-stamp.json), then the
    // dev/local build output (apps/desktop/build/install-stamp.json) so
    // someone running `npm run start` after a local `npm run build` also
    // sees a stamp without needing a packaged build.
    const candidates = [
      process.resourcesPath ? path.join(process.resourcesPath, 'install-stamp.json') : null,
      path.join(APP_ROOT, 'build', 'install-stamp.json')
    ].filter(Boolean)

    for (const p of candidates) {
      try {
        const raw = fs.readFileSync(p, 'utf8')
        const parsed = JSON.parse(raw)

        if (parsed && typeof parsed === 'object' && typeof parsed.commit === 'string' && parsed.commit.length >= 7) {
          if (parsed.schemaVersion !== INSTALL_STAMP_SCHEMA_VERSION) {
            console.warn(
              `[hermes] install-stamp.json schemaVersion ${parsed.schemaVersion} != expected ${INSTALL_STAMP_SCHEMA_VERSION}; ignoring`
            )

            continue
          }

          return Object.freeze({
            schemaVersion: parsed.schemaVersion,
            commit: parsed.commit,
            branch: parsed.branch || null,
            builtAt: parsed.builtAt || null,
            dirty: Boolean(parsed.dirty),
            source: parsed.source || null,
            path: p
          })
        }
      } catch (e) {
        console.warn(`[hermes] install-stamp.json found at ${p} , but parsing failed with ${e}`)
        // Either ENOENT or malformed JSON; try the next candidate
      }
    }

    return null
  }

  // HERMES_HOME — the user-facing root for everything Hermes-related. Mirrors
  // scripts/install.ps1's $HermesHome and scripts/install.sh's $HERMES_HOME.
  //
  // Defaults:
  //   Windows: %LOCALAPPDATA%\hermes (matches install.ps1)
  //   macOS / Linux: ~/.hermes (matches install.sh)
  //
  // Special case for Windows: if the user has a legacy ~/.hermes directory
  // (e.g., from a prior pip install or a manual setup) AND no
  // %LOCALAPPDATA%\hermes yet, prefer the legacy path so we don't orphan their
  // existing config / sessions / .env. New installs go to %LOCALAPPDATA%.
  //
  // HERMES_DESKTOP_USER_DATA_DIR (used by test:desktop:fresh) puts the sandbox
  // HERMES_HOME beneath the throwaway userData dir so a fresh-install run never
  // touches the user's real ~/.hermes / %LOCALAPPDATA%\hermes.
  function resolveHermesHome() {
    if (process.env.HERMES_HOME) {
      return normalizeHermesHomeRoot(process.env.HERMES_HOME)
    }

    if (USER_DATA_OVERRIDE) {
      return path.join(path.resolve(USER_DATA_OVERRIDE), 'hermes-home')
    }

    if (IS_WINDOWS) {
      // A GUI app launched from Explorer inherits the environment block captured
      // at login, so a HERMES_HOME set via `setx` AFTER login is invisible in
      // process.env even though the CLI (a fresh shell) sees it. Without this the
      // backend silently falls back to %LOCALAPPDATA%\hermes and reports "No
      // inference provider configured" despite a valid configured home (#45471).
      // Consult the live User-scoped registry value before the default below.
      const fromRegistry = readWindowsUserEnvVar('HERMES_HOME')

      if (fromRegistry) {
        return normalizeHermesHomeRoot(fromRegistry)
      }
    }

    if (IS_WINDOWS && process.env.LOCALAPPDATA) {
      const localappdata = path.join(process.env.LOCALAPPDATA, 'hermes')
      const legacy = path.join(app.getPath('home'), '.hermes')

      // Migrate transparently to LOCALAPPDATA, but honour an existing legacy
      // ~/.hermes setup (no LOCALAPPDATA install yet) so users don't lose state.
      if (!directoryExists(localappdata) && directoryExists(legacy)) {
        return legacy
      }

      return localappdata
    }

    return path.join(app.getPath('home'), '.hermes')
  }

  return { loadInstallStamp, resolveHermesHome }
}

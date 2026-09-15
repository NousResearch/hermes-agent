import path from 'node:path'

// macOS apps launched from Finder/Dock inherit only /usr/bin:/bin:/usr/sbin:/sbin,
// which misses Homebrew and user-installed CLI tools (codex, git credential
// helpers). Hermes' own managed tools need no PATH help — the backend composes
// their environment in-process via pm — but user tools on PATH do.
const POSIX_SANE_PATH_ENTRIES = Object.freeze([
  '/opt/homebrew/bin',
  '/opt/homebrew/sbin',
  '/usr/local/sbin',
  '/usr/local/bin',
  '/usr/sbin',
  '/usr/bin',
  '/sbin',
  '/bin'
])

function delimiterForPlatform(platform = process.platform) {
  return platform === 'win32' ? ';' : ':'
}

function pathModuleForPlatform(platform = process.platform) {
  return platform === 'win32' ? path.win32 : path.posix
}

function pathEnvKey(env = process.env, platform = process.platform) {
  if (platform !== 'win32') {
    return 'PATH'
  }

  return Object.keys(env || {}).find(key => key.toUpperCase() === 'PATH') || 'PATH'
}

function appendUniquePathEntries(entries, { delimiter = path.delimiter } = {}) {
  const seen = new Set()
  const ordered = []

  for (const entry of entries) {
    if (!entry) {
      continue
    }

    const parts = Array.isArray(entry) ? entry : String(entry).split(delimiter)

    for (const part of parts) {
      if (!part || seen.has(part)) {
        continue
      }

      seen.add(part)
      ordered.push(part)
    }
  }

  return ordered.join(delimiter)
}

function normalizeHermesHomeRoot(hermesHome, { pathModule = pathModuleForPlatform(process.platform) }: any = {}) {
  if (!hermesHome) {
    return hermesHome
  }

  const resolved = pathModule.resolve(String(hermesHome))
  const parent = pathModule.dirname(resolved)

  if (pathModule.basename(parent).toLowerCase() === 'profiles') {
    return pathModule.dirname(parent)
  }

  return resolved
}

/**
 * The environment for the spawned Python backend. Electron knows ONE thing:
 * where the interpreter is (by convention). Everything else — managed tool
 * PATHs, browser paths, node — is composed in-process by pm when the backend
 * spawns tools. PYTHONPATH/PYTHONHOME are scrubbed so an inherited value
 * can't make the backend import modules from another checkout.
 */
function buildDesktopBackendEnv({ currentEnv = process.env, platform = process.platform }: any = {}) {
  const delimiter = delimiterForPlatform(platform)
  const key = pathEnvKey(currentEnv, platform)
  const saneEntries = platform === 'win32' ? [] : POSIX_SANE_PATH_ENTRIES

  return {
    PYTHONPATH: '',
    PYTHONHOME: '',
    // Force PEP 540 UTF-8 mode in the spawned Python backend so its stdio and
    // subprocess defaults are UTF-8 even on non-UTF-8 Windows locales (GBK,
    // cp1252, ...). hermes_bootstrap sets this inside the child too, but only
    // after import — anything emitted earlier (interpreter startup errors,
    // pre-bootstrap tracebacks) still decodes with the locale default without
    // this. User's explicit setting wins. Re-port of PR #56499 (echoriver89).
    PYTHONUTF8: currentEnv?.PYTHONUTF8 ?? '1',
    [key]: appendUniquePathEntries([currentEnv?.[key] || '', saneEntries], { delimiter })
  }
}

export {
  appendUniquePathEntries,
  buildDesktopBackendEnv,
  delimiterForPlatform,
  normalizeHermesHomeRoot,
  pathEnvKey,
  POSIX_SANE_PATH_ENTRIES
}

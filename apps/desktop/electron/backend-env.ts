import fs from 'node:fs'
import os from 'node:os'
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

function resolveHermesHomePath(hermesHome, { pathModule, homedir = os.homedir() }: any) {
  // fish (and any shell when the value is quoted) hands a literal `~` through; path.resolve()
  // would pin it under cwd and the Python backend inherits that absolute path via HERMES_HOME.
  let raw = String(hermesHome)

  if (raw === '~' || raw.startsWith('~/') || (pathModule === path.win32 && raw.startsWith('~\\'))) {
    raw = pathModule.join(homedir, raw.slice(1))
  }

  return pathModule.resolve(raw)
}

function isProfileHome(resolved, pathModule) {
  return pathModule.basename(pathModule.dirname(resolved)).toLowerCase() === 'profiles'
}

function normalizeHermesHomeRoot(
  hermesHome,
  { pathModule = pathModuleForPlatform(process.platform), homedir = os.homedir() }: any = {}
) {
  if (!hermesHome) {
    return hermesHome
  }

  const resolved = resolveHermesHomePath(hermesHome, { pathModule, homedir })

  return isProfileHome(resolved, pathModule) ? pathModule.dirname(pathModule.dirname(resolved)) : resolved
}

// OS/interpreter names a dotenv may redeclare that no child can run without.
const PROCESS_ENV_NAMES = new Set([
  'APPDATA',
  'COMSPEC',
  'HERMES_HOME',
  'HOME',
  'LANG',
  'LC_ALL',
  'LOCALAPPDATA',
  'PATH',
  'PWD',
  'PYTHONPATH',
  'SHELL',
  'SSL_CERT_FILE',
  'SYSTEMROOT',
  'TEMP',
  'TMP',
  'TMPDIR',
  'TZ',
  'USER',
  'USERPROFILE',
  'VIRTUAL_ENV'
])

function dotenvKeyNames(contents = '') {
  return String(contents)
    .replace(/^\uFEFF/, '')
    .split(/\r?\n/)
    .flatMap(line => line.match(/^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=/)?.slice(1, 2) ?? [])
}

function readTextOrEmpty(fsModule, file) {
  try {
    return String(fsModule.readFileSync(file, 'utf8'))
  } catch {
    return ''
  }
}

/**
 * Parent env for a local `hermes serve` child of `profile` (#68367).
 *
 * `hermes desktop` loads its launch profile's `.env`/`.op.env` into os.environ
 * before exec'ing Electron, so `process.env` carries that profile's platform
 * credentials. A child for ANOTHER profile would inherit them ahead of its own
 * dotenv (`.op.env` is even skipped once OP_SERVICE_ACCOUNT_TOKEN is set) and,
 * e.g., connect the same Tlon ship as the default gateway. Drop every name the
 * launch profile's dotenv declares, as `_profile_action_environment` does for
 * dashboard actions; the child reloads its own scope. The launch profile's own
 * backend keeps the env unchanged, and shell exports the launch dotenv never
 * declared pass through everywhere.
 *
 * `profile` null/empty means no `--profile` flag: the child follows the sticky
 * `active_profile` like a bare `hermes serve` (`_apply_profile_override`).
 */
function profileBackendParentEnv({
  hermesHome,
  profile,
  currentEnv = process.env,
  platform = process.platform,
  fsModule = fs,
  pathModule = pathModuleForPlatform(platform)
}: any = {}) {
  const env = { ...(currentEnv || {}) }

  if (!hermesHome) {
    return env
  }

  const fold = platform === 'win32' ? (value: string) => value.toUpperCase() : (value: string) => value
  const inheritedHome = currentEnv?.HERMES_HOME ? resolveHermesHomePath(currentEnv.HERMES_HOME, { pathModule }) : null
  const launchHome = inheritedHome && isProfileHome(inheritedHome, pathModule) ? inheritedHome : hermesHome
  const name = profile || readTextOrEmpty(fsModule, pathModule.join(hermesHome, 'active_profile')).trim()
  const targetHome = !name || name === 'default' ? hermesHome : pathModule.join(hermesHome, 'profiles', name)

  if (fold(pathModule.resolve(launchHome)) === fold(pathModule.resolve(targetHome))) {
    return env
  }

  const launchOwned = new Set(
    ['.env', '.op.env']
      .flatMap(file => dotenvKeyNames(readTextOrEmpty(fsModule, pathModule.join(launchHome, file))))
      .filter(key => !PROCESS_ENV_NAMES.has(key.toUpperCase()))
      .map(fold)
  )

  for (const key of Object.keys(env)) {
    if (launchOwned.has(fold(key))) {
      delete env[key]
    }
  }

  return env
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

  const homeEnv: NodeJS.ProcessEnv = {}

  if (platform === 'win32') {
    const value = (name: string): string | undefined =>
      currentEnv?.[Object.keys(currentEnv || {}).find(key => key.toUpperCase() === name) ?? name]

    // Unlike Node's OS-backed homedir(), Python's Path.home() needs these
    // environment variables. Registry-only GUI launches omit them (#122384).
    // Preserve explicit paths (including drive/path pairs) and only backfill a
    // missing profile; never infer the account home from HERMES_HOME.
    if (!value('USERPROFILE') && !(value('HOMEDRIVE') && value('HOMEPATH'))) {
      try {
        const home = os.homedir()

        if (path.win32.isAbsolute(home)) {
          const key = Object.keys(currentEnv || {}).find(key => key.toUpperCase() === 'USERPROFILE') ?? 'USERPROFILE'
          homeEnv[key] = home
        }
      } catch {
        // A failed OS lookup must retain the existing probe/fallback behavior.
      }
    }
  }

  return {
    ...homeEnv,
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
  POSIX_SANE_PATH_ENTRIES,
  profileBackendParentEnv
}

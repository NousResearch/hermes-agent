import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

// Match the POSIX fallback surface used by the Python terminal environment.
// macOS apps launched from Finder/Dock often inherit only /usr/bin:/bin:/usr/sbin:/sbin,
// which misses Apple Silicon Homebrew and user-installed CLI tools such as codex.
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

function currentPathValue(env = process.env, platform = process.platform) {
  const key = pathEnvKey(env, platform)

  return env?.[key] || ''
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

/**
 * Hermes-managed Node.js directories, in preferred lookup order.
 *
 * There are two on-disk layouts. `scripts/install.ps1` unpacks portable Node
 * straight into `%LOCALAPPDATA%\hermes\node` (node.exe at the root, no `bin\`);
 * `scripts/install.sh` and the node-bootstrap helper use the POSIX
 * `$HERMES_HOME/node/bin`. Emit BOTH on every platform so mixed and migrated
 * installs resolve, leading with the layout native to the current platform.
 *
 * This is the single source of truth for the ordering rule on the Node side —
 * `main.ts` imports it rather than keeping its own copy. Mirrors
 * `iter_hermes_node_dirs()` in hermes_constants.py, which the Electron main
 * process cannot import.
 */
function hermesManagedNodePathEntries(
  hermesHome,
  { platform = process.platform, pathModule = pathModuleForPlatform(platform) }: any = {}
) {
  if (!hermesHome) {
    return []
  }

  const root = pathModule.join(hermesHome, 'node')
  const bin = pathModule.join(root, 'bin')

  return platform === 'win32' ? [root, bin] : [bin, root]
}

function buildDesktopBackendPath({
  hermesHome,
  venvRoot,
  currentPath = '',
  platform = process.platform,
  pathModule = pathModuleForPlatform(platform)
}: any = {}) {
  const delimiter = delimiterForPlatform(platform)
  const hermesNodeDirs = hermesManagedNodePathEntries(hermesHome, { platform, pathModule })
  const venvBin = venvRoot ? pathModule.join(venvRoot, platform === 'win32' ? 'Scripts' : 'bin') : null
  const saneEntries = platform === 'win32' ? [] : POSIX_SANE_PATH_ENTRIES

  return appendUniquePathEntries([hermesNodeDirs, venvBin, currentPath, saneEntries], { delimiter })
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

function buildDesktopBackendEnv({
  hermesHome,
  pythonPathEntries = [],
  venvRoot,
  currentEnv = process.env,
  platform = process.platform,
  pathModule = pathModuleForPlatform(platform)
}: any = {}) {
  const delimiter = delimiterForPlatform(platform)
  const currentPythonPath = currentEnv?.PYTHONPATH || ''
  const key = pathEnvKey(currentEnv, platform)

  return {
    PYTHONPATH: appendUniquePathEntries([...pythonPathEntries, currentPythonPath], { delimiter }),
    // Force PEP 540 UTF-8 mode in the spawned Python backend so its stdio and
    // subprocess defaults are UTF-8 even on non-UTF-8 Windows locales (GBK,
    // cp1252, ...). hermes_bootstrap sets this inside the child too, but only
    // after import — anything emitted earlier (interpreter startup errors,
    // pre-bootstrap tracebacks) still decodes with the locale default without
    // this. User's explicit setting wins. Re-port of PR #56499 (echoriver89).
    PYTHONUTF8: currentEnv?.PYTHONUTF8 ?? '1',
    [key]: buildDesktopBackendPath({
      hermesHome,
      venvRoot,
      currentPath: currentPathValue(currentEnv, platform),
      platform,
      pathModule
    })
  }
}

export {
  appendUniquePathEntries,
  buildDesktopBackendEnv,
  buildDesktopBackendPath,
  delimiterForPlatform,
  hermesManagedNodePathEntries,
  normalizeHermesHomeRoot,
  pathEnvKey,
  POSIX_SANE_PATH_ENTRIES,
  profileBackendParentEnv
}

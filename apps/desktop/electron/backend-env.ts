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

function environmentValue(env = process.env, name: string) {
  const key = Object.keys(env || {}).find(key => key.toUpperCase() === name.toUpperCase())

  return key ? env?.[key] : undefined
}

function removeVenvBinFromPath({
  currentPath = '',
  venvRoot,
  platform = process.platform,
  pathModule = pathModuleForPlatform(platform)
}: any = {}) {
  if (!venvRoot) {
    return currentPath
  }

  const delimiter = delimiterForPlatform(platform)

  const venvBin = pathModule.resolve(
    pathModule.join(venvRoot, platform === 'win32' ? 'Scripts' : 'bin')
  )

  const normalize = value => {
    const resolved = pathModule.resolve(String(value))

    return platform === 'win32' ? resolved.toLowerCase() : resolved
  }

  const normalizedVenvBin = normalize(venvBin)

  return String(currentPath)
    .split(delimiter)
    .filter(entry => entry && normalize(entry) !== normalizedVenvBin)
    .join(delimiter)
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

function buildDesktopBackendEnv({
  hermesHome,
  pythonPathEntries = [],
  venvRoot,
  currentEnv = process.env,
  platform = process.platform,
  pathModule = pathModuleForPlatform(platform)
}: any = {}): Record<string, string> {
  const delimiter = delimiterForPlatform(platform)
  const key = pathEnvKey(currentEnv, platform)

  const currentPath = removeVenvBinFromPath({
    currentPath: currentPathValue(currentEnv, platform),
    venvRoot: environmentValue(currentEnv, 'VIRTUAL_ENV'),
    platform,
    pathModule
  })

  return {
    ...(venvRoot ? { VIRTUAL_ENV: venvRoot } : {}),
    // Never carry parent PYTHONPATH into the backend. The selected source root
    // and its owning venv site-packages are supplied explicitly by the
    // resolver; inheriting anything else can mix Python ABIs.
    PYTHONPATH: appendUniquePathEntries(pythonPathEntries, { delimiter }),
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
      currentPath,
      platform,
      pathModule
    })
  }
}

function buildDesktopBackendChildEnv({
  currentEnv = process.env,
  backendEnv = {},
  overrides = {},
  platform = process.platform,
  pathModule = pathModuleForPlatform(platform)
}: any = {}): Record<string, string> {
  const sanitized = { ...currentEnv }
  const inheritedPythonKeys = new Set(['PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV'])
  const inheritedVenvRoot = environmentValue(currentEnv, 'VIRTUAL_ENV')

  for (const key of Object.keys(sanitized)) {
    if (inheritedPythonKeys.has(key.toUpperCase())) {
      delete sanitized[key]
    }
  }

  const key = pathEnvKey(sanitized, platform)
  const currentPath = currentPathValue(sanitized, platform)

  if (currentPath) {
    sanitized[key] = removeVenvBinFromPath({
      currentPath,
      venvRoot: inheritedVenvRoot,
      platform,
      pathModule
    })
  }

  return {
    ...sanitized,
    ...backendEnv,
    ...overrides
  }
}

function buildDesktopPythonBackend({
  root,
  label,
  backendArgs,
  command,
  runtimeVenvRoot,
  sitePackagesEntries = [],
  hermesHome,
  currentEnv = process.env,
  platform = process.platform,
  pathModule = pathModuleForPlatform(platform),
  bootstrap = false
}: any) {
  return {
    kind: 'python' as const,
    label,
    command,
    args: ['-m', 'hermes_cli.main', ...backendArgs],
    env: buildDesktopBackendEnv({
      hermesHome,
      pythonPathEntries: [root, ...sitePackagesEntries],
      venvRoot: runtimeVenvRoot,
      currentEnv,
      platform,
      pathModule
    }),
    root,
    bootstrap,
    shell: false as const
  }
}

export {
  appendUniquePathEntries,
  buildDesktopBackendChildEnv,
  buildDesktopBackendEnv,
  buildDesktopBackendPath,
  buildDesktopPythonBackend,
  delimiterForPlatform,
  hermesManagedNodePathEntries,
  normalizeHermesHomeRoot,
  pathEnvKey,
  POSIX_SANE_PATH_ENTRIES,
  removeVenvBinFromPath
}

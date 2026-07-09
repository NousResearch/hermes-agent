const path = require('node:path')

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
  if (platform !== 'win32') return 'PATH'
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
    if (!entry) continue
    const parts = Array.isArray(entry) ? entry : String(entry).split(delimiter)
    for (const part of parts) {
      if (!part || seen.has(part)) continue
      seen.add(part)
      ordered.push(part)
    }
  }

  return ordered.join(delimiter)
}

function buildDesktopBackendPath({
  hermesHome,
  venvRoot,
  currentPath = '',
  platform = process.platform,
  pathModule = pathModuleForPlatform(platform)
} = {}) {
  const delimiter = delimiterForPlatform(platform)
  const hermesNodeBin = hermesHome ? pathModule.join(hermesHome, 'node', 'bin') : null
  const venvBin = venvRoot ? pathModule.join(venvRoot, platform === 'win32' ? 'Scripts' : 'bin') : null
  const saneEntries = platform === 'win32' ? [] : POSIX_SANE_PATH_ENTRIES

  return appendUniquePathEntries([hermesNodeBin, venvBin, currentPath, saneEntries], { delimiter })
}

function normalizeHermesHomeRoot(hermesHome, { pathModule = pathModuleForPlatform(process.platform) } = {}) {
  if (!hermesHome) return hermesHome
  const resolved = pathModule.resolve(String(hermesHome))
  const parent = pathModule.dirname(resolved)
  if (pathModule.basename(parent).toLowerCase() === 'profiles') {
    return pathModule.dirname(parent)
  }
  return resolved
}

function isPathUnderRoot(candidate, root, { pathModule = pathModuleForPlatform(process.platform), caseSensitive = true } = {}) {
  if (!candidate || !root) return false
  let resolvedCandidate = pathModule.resolve(String(candidate))
  let resolvedRoot = pathModule.resolve(String(root))
  if (!caseSensitive) {
    resolvedCandidate = resolvedCandidate.toLowerCase()
    resolvedRoot = resolvedRoot.toLowerCase()
  }
  const rootPrefix = resolvedRoot.endsWith(pathModule.sep) ? resolvedRoot : `${resolvedRoot}${pathModule.sep}`
  return resolvedCandidate === resolvedRoot || resolvedCandidate.startsWith(rootPrefix)
}

function isEphemeralPath(candidate, {
  platform = process.platform,
  pathModule = pathModuleForPlatform(platform),
  tempRoots
} = {}) {
  if (!candidate) return false
  const roots = tempRoots || (platform === 'win32'
    ? [process.env.TEMP, process.env.TMP]
    : ['/tmp', '/private/tmp', '/var/tmp'])
  const caseSensitive = platform !== 'win32'
  return roots.filter(Boolean).some(root => isPathUnderRoot(candidate, root, { pathModule, caseSensitive }))
}

function packagedHermesHomeFromUserData(userDataDir, { pathModule = pathModuleForPlatform(process.platform) } = {}) {
  if (!userDataDir) return userDataDir
  return pathModule.join(pathModule.resolve(String(userDataDir)), 'hermes-home')
}

function buildDesktopBackendEnv({
  hermesHome,
  pythonPathEntries = [],
  venvRoot,
  currentEnv = process.env,
  platform = process.platform,
  pathModule = pathModuleForPlatform(platform)
} = {}) {
  const delimiter = delimiterForPlatform(platform)
  const currentPythonPath = currentEnv?.PYTHONPATH || ''
  const key = pathEnvKey(currentEnv, platform)

  return {
    PYTHONPATH: appendUniquePathEntries([...pythonPathEntries, currentPythonPath], { delimiter }),
    [key]: buildDesktopBackendPath({
      hermesHome,
      venvRoot,
      currentPath: currentPathValue(currentEnv, platform),
      platform,
      pathModule
    })
  }
}

module.exports = {
  POSIX_SANE_PATH_ENTRIES,
  appendUniquePathEntries,
  buildDesktopBackendEnv,
  buildDesktopBackendPath,
  delimiterForPlatform,
  isEphemeralPath,
  normalizeHermesHomeRoot,
  packagedHermesHomeFromUserData,
  pathEnvKey
}

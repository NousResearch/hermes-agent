import fs from 'node:fs'
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

function dotenvKeyNames(contents = '') {
  const names = []

  for (const line of String(contents).split(/\r?\n/)) {
    const match = line.match(/^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=/)

    if (match) {
      names.push(match[1])
    }
  }

  return names
}

function buildProfileScopedParentEnv({
  currentEnv = process.env,
  profileEnvContents = [],
  platform = process.platform
}: any = {}) {
  const normalizeKey = platform === 'win32' ? key => key.toUpperCase() : key => key

  const profileOwnedKeys = new Set(
    profileEnvContents.flatMap(contents => dotenvKeyNames(contents)).map(key => normalizeKey(key))
  )

  return Object.fromEntries(
    Object.entries(currentEnv || {}).filter(([key]) => !profileOwnedKeys.has(normalizeKey(key)))
  )
}

function readProfileEnvContents(hermesHome, { fsModule = fs, pathModule = path }: any = {}) {
  const envPaths = [pathModule.join(hermesHome, '.env')]
  const profilesRoot = pathModule.join(hermesHome, 'profiles')

  try {
    for (const entry of fsModule.readdirSync(profilesRoot, { withFileTypes: true })) {
      if (entry.isDirectory()) {
        envPaths.push(pathModule.join(profilesRoot, entry.name, '.env'))
      }
    }
  } catch {
    // A missing profiles directory is the normal single-profile case.
  }

  return envPaths.flatMap(envPath => {
    try {
      return [fsModule.readFileSync(envPath, 'utf8')]
    } catch {
      return []
    }
  })
}

function buildProfileBackendParentEnv({
  hermesHome,
  profile,
  currentEnv = process.env,
  platform = process.platform,
  fsModule = fs,
  pathModule = path
}: any = {}) {
  if (!profile || !hermesHome) {
    return { ...(currentEnv || {}) }
  }

  return buildProfileScopedParentEnv({
    currentEnv,
    profileEnvContents: readProfileEnvContents(hermesHome, { fsModule, pathModule }),
    platform
  })
}

function buildDesktopBackendPath({
  hermesHome,
  venvRoot,
  currentPath = '',
  platform = process.platform,
  pathModule = pathModuleForPlatform(platform)
}: any = {}) {
  const delimiter = delimiterForPlatform(platform)
  const hermesNodeBin = hermesHome ? pathModule.join(hermesHome, 'node', 'bin') : null
  const venvBin = venvRoot ? pathModule.join(venvRoot, platform === 'win32' ? 'Scripts' : 'bin') : null
  const saneEntries = platform === 'win32' ? [] : POSIX_SANE_PATH_ENTRIES

  return appendUniquePathEntries([hermesNodeBin, venvBin, currentPath, saneEntries], { delimiter })
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
}: any = {}) {
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

export {
  appendUniquePathEntries,
  buildDesktopBackendEnv,
  buildDesktopBackendPath,
  buildProfileBackendParentEnv,
  buildProfileScopedParentEnv,
  delimiterForPlatform,
  dotenvKeyNames,
  normalizeHermesHomeRoot,
  pathEnvKey,
  POSIX_SANE_PATH_ENTRIES,
  readProfileEnvContents
}

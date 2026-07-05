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

// hc-406: HuggingFace mirror for mainland China. faster-whisper (local STT,
// the default speech-to-text provider) lazily downloads its ~150 MB Whisper
// model from the Hub on first voice message; huggingface.co is blocked/slow in
// CN, so a bare-network desktop hangs the whole transcription. huggingface_hub
// (a faster-whisper transitive dep) resolves the Hub host from HF_ENDPOINT, so
// exporting this into the backend subprocess env transparently routes every Hub
// download (STT model, marker-pdf OCR models, the huggingface-hub skill's `hf`
// CLI, …) through the official transparent CN mirror. Overridable: we only set
// it when the parent env hasn't already (a power user / staging can point at
// ModelScope or the real Hub). The runtime never reads HF_ENDPOINT itself; this
// is purely for the Python libs it shells into.
const HF_MIRROR_ENDPOINT = 'https://hf-mirror.com'

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

  return appendUniquePathEntries(
    [hermesNodeBin, venvBin, currentPath, saneEntries],
    { delimiter }
  )
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

  const env = {
    PYTHONPATH: appendUniquePathEntries([...pythonPathEntries, currentPythonPath], { delimiter }),
    [key]: buildDesktopBackendPath({
      hermesHome,
      venvRoot,
      currentPath: currentPathValue(currentEnv, platform),
      platform,
      pathModule
    })
  }

  // hc-406: seed the CN HuggingFace mirror for the Python Hub-download path
  // (faster-whisper STT model, marker-pdf OCR, the `hf` CLI). Add-only: never
  // clobber an HF_ENDPOINT the parent env already set (staging / power-user /
  // ModelScope override). The spawn merges `{ ...process.env, ...backend.env }`,
  // so a value here wins over inheritance — hence the explicit passthrough of an
  // existing value so an override survives.
  const existingHfEndpoint = String(currentEnv?.HF_ENDPOINT || '').trim()
  env.HF_ENDPOINT = existingHfEndpoint || HF_MIRROR_ENDPOINT

  return env
}

module.exports = {
  HF_MIRROR_ENDPOINT,
  POSIX_SANE_PATH_ENTRIES,
  appendUniquePathEntries,
  buildDesktopBackendEnv,
  buildDesktopBackendPath,
  delimiterForPlatform,
  normalizeHermesHomeRoot,
  pathEnvKey
}

const path = require('node:path')

// resolveBackendRoot — pick the tree the desktop app's Python backend should run from (SPEC §4.1).
// Mirrors resolveUpdateRoot()'s precedence, but for the BACKEND interpreter/import source, and prefers the
// RUNTIME deploy tree (reviewed, FF-only) over the DEV tree — matching what every gateway already does.
//
// Precedence:
//   1. override      — config.yaml desktop.backend_root (bridged via HERMES_DESKTOP_BACKEND_ROOT), if it is a
//                      usable hermes source root WITH a venv. A set-but-unusable value → fall through (a typo
//                      can't brick the app).
//   2. runtime tree  — <hermesHome>/runtime/hermes-agent, if it is a hermes source root WITH a venv. DEFAULT.
//   3. dev tree      — <hermesHome>/hermes-agent (ACTIVE_HERMES_ROOT). Fallback (fresh/CLI-only installs, or
//                      a broken/absent runtime tree).
//
// Pure + dependency-injected so it unit-tests without launching Electron:
//   deps = { isSourceRoot(root)->bool, hasVenv(root)->bool, onFallback(reason, root)->void }
// Returns { root, venvRoot, tier }.
function resolveBackendRoot({
  hermesHome,
  activeRoot,               // ACTIVE_HERMES_ROOT (dev tree)
  override,                 // resolved HERMES_DESKTOP_BACKEND_ROOT (already config-read + env), or falsy
  isSourceRoot,
  hasVenv,
  onFallback = () => {},
  pathModule = path,
} = {}) {
  const runtimeRoot = hermesHome ? pathModule.join(hermesHome, 'runtime', 'hermes-agent') : null
  const venvOf = (root) => (root ? pathModule.join(root, 'venv') : null)
  const usable = (root) => Boolean(root) && isSourceRoot(root) && hasVenv(root)

  // Tier 1: explicit override.
  if (override) {
    const resolved = pathModule.resolve(String(override))
    if (usable(resolved)) return { root: resolved, venvRoot: venvOf(resolved), tier: 'override' }
    onFallback('override-unusable', resolved) // set but not a usable source-root+venv → fall through
  }

  // Tier 2: runtime deploy tree (default).
  if (usable(runtimeRoot)) return { root: runtimeRoot, venvRoot: venvOf(runtimeRoot), tier: 'runtime' }

  // Tier 3: dev tree fallback. Surface WHY we fell back (absent runtime tree vs unusable) — one shared path.
  if (runtimeRoot) onFallback('runtime-unavailable', runtimeRoot)
  return { root: activeRoot, venvRoot: venvOf(activeRoot), tier: 'dev' }
}

module.exports = { resolveBackendRoot }

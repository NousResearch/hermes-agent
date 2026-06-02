'use strict'

/**
 * Small, dependency-injected helpers for deciding whether Desktop can reuse an
 * already-installed Hermes Agent runtime. Keep this separate from main.cjs so
 * the decision is unit-testable without booting Electron.
 */
function hasUsableActiveInstall(opts) {
  const {
    activeRoot,
    canImportHermesCli,
    fileExists,
    getVenvPython,
    isHermesSourceRoot,
    rememberLog,
    venvRoot
  } = opts || {}

  if (!activeRoot || !venvRoot) return false
  if (typeof isHermesSourceRoot !== 'function' || typeof fileExists !== 'function') return false
  if (typeof getVenvPython !== 'function' || typeof canImportHermesCli !== 'function') return false

  if (!isHermesSourceRoot(activeRoot)) return false

  const venvPython = getVenvPython(venvRoot)
  if (!venvPython || !fileExists(venvPython)) return false

  if (!canImportHermesCli(venvPython)) {
    if (typeof rememberLog === 'function') {
      rememberLog(`Ignoring existing Hermes install at ${activeRoot}: hermes_cli is not importable from ${venvPython}.`)
    }
    return false
  }

  return true
}

module.exports = { hasUsableActiveInstall }

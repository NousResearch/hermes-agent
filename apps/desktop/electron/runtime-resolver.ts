type ImportProbe = (
  pythonPath: string,
  opts?: { env?: Record<string, string> }
) => boolean

type ActiveInstallOptions = {
  activeRoot: string
  venvRoot: string
  getVenvPython: (venvRoot: string) => string
  isHermesSourceRoot: (activeRoot: string) => boolean
  fileExists: (candidate: string) => boolean
  canImportHermesCli: ImportProbe
  existingPythonPath?: string
  pathDelimiter: string
}

/**
 * Return whether the canonical source checkout and venv are runnable without
 * relying on Desktop's bootstrap marker. The import probe intentionally runs
 * on every call so an in-place dependency or venv regression cannot be hidden
 * by stale resolver state.
 */
function hasUsableActiveInstall(options: ActiveInstallOptions): boolean {
  const {
    activeRoot,
    venvRoot,
    getVenvPython,
    isHermesSourceRoot,
    fileExists,
    canImportHermesCli,
    existingPythonPath,
    pathDelimiter
  } = options

  if (!activeRoot || !venvRoot || !isHermesSourceRoot(activeRoot)) return false

  const venvPython = getVenvPython(venvRoot)
  if (!venvPython || !fileExists(venvPython)) return false

  return canImportHermesCli(venvPython, {
    env: {
      PYTHONPATH: [activeRoot, existingPythonPath].filter(Boolean).join(pathDelimiter)
    }
  })
}

export { hasUsableActiveInstall }
export type { ActiveInstallOptions }

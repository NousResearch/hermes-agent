const path = require('node:path')

/** True when `dir` lives inside a packaged app bundle / install tree. */
function isPackagedInstallPath(dir, { installRoots, isPackaged }) {
  if (!isPackaged || !dir) {
    return false
  }

  let resolved

  try {
    resolved = path.resolve(String(dir))
  } catch {
    return false
  }

  const roots = new Set((installRoots ?? []).filter(Boolean).map(candidate => path.resolve(String(candidate))))

  for (const root of roots) {
    if (resolved === root) {
      return true
    }

    const rel = path.relative(root, resolved)

    if (rel && !rel.startsWith('..') && !path.isAbsolute(rel)) {
      return true
    }
  }

  return false
}

module.exports = { isPackagedInstallPath }

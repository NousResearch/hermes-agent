import path from 'node:path'

/** True when `dir` lives inside a packaged app bundle / install tree. */
function isPackagedInstallPath(dir, { installRoots, isPackaged }: { installRoots: string[]; isPackaged: boolean }) {
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

    const rel = path.relative(root, resolved) as any

    if (rel && !rel.startsWith('..') && !path.isAbsolute(rel)) {
      return true
    }
  }

  return false
}

/** True when `dir` is `root` itself or nested under it. */
function isAtOrUnderRoot(dir, root) {
  let resolvedDir
  let resolvedRoot

  try {
    resolvedDir = path.resolve(String(dir))
    resolvedRoot = path.resolve(String(root))
  } catch {
    return false
  }

  if (resolvedDir === resolvedRoot) {
    return true
  }

  const rel = path.relative(resolvedRoot, resolvedDir)

  return Boolean(rel) && !rel.startsWith('..') && !path.isAbsolute(rel)
}

/**
 * The TERMINAL_CWD pin for a spawned backend. The spawn cwd may be the
 * dev-run source-tree fallback (a source run with no chosen project dir
 * lands in the repo); pinning that verbatim makes every session's tool cwd
 * the install tree and defeats the agent's install-tree context guard, so it
 * falls back to the home dir unless the user deliberately chose the tree
 * (see #121259). Packaged spawns pass through (their install-path guard
 * already runs in the candidate chain).
 *
 * ponytail: at-or-inside string check, not a realpath comparison — symlinked
 * checkouts resolve the same way on both sides via path.resolve.
 */
function resolveTerminalCwd(spawnCwd, { chosenCwd, homeDir, isPackaged, sourceRoot }) {
  if (isPackaged || !spawnCwd || !sourceRoot || !homeDir) {
    return spawnCwd
  }

  try {
    if (chosenCwd && path.resolve(String(chosenCwd)) === path.resolve(String(spawnCwd))) {
      return spawnCwd
    }
  } catch {
    return spawnCwd
  }

  return isAtOrUnderRoot(spawnCwd, sourceRoot) ? homeDir : spawnCwd
}

export { isPackagedInstallPath, resolveTerminalCwd }

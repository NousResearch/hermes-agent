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

interface WorkspaceCwdCandidate {
  cwd: null | string | undefined
  source?: 'launch'
}

/** Validate the existing cwd ladder while retaining explicit terminal-launch intent. */
export function resolveWorkspaceCwd(
  candidates: WorkspaceCwdCandidate[],
  home: string,
  acceptsDirectory: (cwd: string) => boolean
): { cwd: string; launchCwd: string } {
  for (const candidate of candidates) {
    if (!candidate.cwd) {
      continue
    }

    const cwd = path.resolve(candidate.cwd)

    if (acceptsDirectory(cwd)) {
      return { cwd, launchCwd: candidate.source === 'launch' ? cwd : '' }
    }
  }

  return { cwd: home, launchCwd: '' }
}

export { isPackagedInstallPath }

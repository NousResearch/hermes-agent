import os from 'node:os'
import path from 'node:path'

export interface WorkingDirectoryFacts {
  platform: NodeJS.Platform
  isPackaged: boolean
  execPath: string
  cwd: string
  homedir: string
}

/**
 * Where a packaged Windows app must move its CWD before main spawns anything, or null to stay.
 * Pure (the host is data) so every CI host checks the decision; the live EBUSY proof runs on Windows.
 */
export function packagedWorkingDirectoryTarget(facts: WorkingDirectoryFacts): string | null {
  if (facts.platform !== 'win32' || !facts.isPackaged) {
    return null
  }

  const relative = path.win32.relative(path.win32.dirname(facts.execPath), facts.cwd)

  // Preserve an explicitly supplied workspace outside the install directory.
  if (relative === '..' || relative.startsWith(`..${path.win32.sep}`) || path.win32.isAbsolute(relative)) {
    return null
  }

  return facts.homedir
}

/** Keep resident children from inheriting a CWD that pins the replaceable app tree. */
export function releasePackagedWorkingDirectory(isPackaged: boolean): void {
  const target = packagedWorkingDirectoryTarget({
    platform: process.platform,
    isPackaged,
    execPath: process.execPath,
    cwd: process.cwd(),
    homedir: os.homedir()
  })

  if (target) {
    process.chdir(target)
  }
}

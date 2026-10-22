import os from 'node:os'
import path from 'node:path'

/** Keep resident children from inheriting a CWD that pins the replaceable app tree. */
export function releasePackagedWorkingDirectory(isPackaged: boolean): void {
  if (process.platform !== 'win32' || !isPackaged) {
    return
  }

  const relative = path.relative(path.dirname(process.execPath), process.cwd())

  // Preserve an explicitly supplied workspace outside the install directory.
  if (relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) {
    return
  }

  process.chdir(os.homedir())
}

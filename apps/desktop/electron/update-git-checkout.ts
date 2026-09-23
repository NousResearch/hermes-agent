import fs from 'node:fs'
import path from 'node:path'

/** Return whether root is a Git checkout the Desktop updater can operate on. */
export function isGitCheckout(root: string): boolean {
  try {
    const stat = fs.statSync(path.join(root, '.git'))

    return stat.isDirectory() || stat.isFile()
  } catch {
    return false
  }
}

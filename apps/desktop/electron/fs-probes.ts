// Synchronous filesystem probes main.ts uses on its boot and update paths.

import fs from 'node:fs'
import path from 'node:path'

export function fileExists(filePath: string): boolean {
  try {
    return fs.statSync(filePath).isFile()
  } catch {
    return false
  }
}

export function isGitCheckout(root: string): boolean {
  return fs.existsSync(path.join(root, '.git'))
}

export function directoryExists(filePath: string): boolean {
  try {
    return fs.statSync(filePath).isDirectory()
  } catch {
    return false
  }
}

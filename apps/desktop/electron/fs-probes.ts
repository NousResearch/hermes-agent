import fs from 'node:fs'

// Synchronous existence probes shared by the main process; a missing or
// unreadable path is simply "not there", never an error.

export function fileExists(filePath: string): boolean {
  try {
    return fs.statSync(filePath).isFile()
  } catch {
    return false
  }
}

export function directoryExists(filePath: string): boolean {
  try {
    return fs.statSync(filePath).isDirectory()
  } catch {
    return false
  }
}

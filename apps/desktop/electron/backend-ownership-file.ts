import fs from 'node:fs'

export function readBackendOwnershipFile(filePath: string): string | null {
  try {
    return fs.readFileSync(filePath, 'utf8')
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      return null
    }

    // An unreadable roster is not empty: callers must not overwrite its records.
    throw error
  }
}

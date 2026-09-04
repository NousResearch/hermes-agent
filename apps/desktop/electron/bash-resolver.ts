import { accessSync, constants, statSync } from 'node:fs'

// Resolve a bash interpreter for spawning scripts on POSIX systems.
//
// `/bin/bash` does not exist on NixOS, musl distros, or minimal containers.
// Resolution order:
//   1. explicit override (deps.override)
//   2. `bash` on PATH
//   3. well-known absolute locations
// Returns the first candidate that exists, or null when nothing does. Callers
// decide what null means for their context (skip, fall back to sh, surface).
export function resolveBashExecutable(deps: any = {}) {
  const candidates: string[] = []

  if (typeof deps.override === 'string' && deps.override.trim()) {
    candidates.push(deps.override.trim())
  }

  const pathValue = deps.pathEnv ?? process.env.PATH

  if (typeof pathValue === 'string' && pathValue.trim()) {
    for (const dir of pathValue.split(deps.pathDelimiter ?? ':')) {
      if (dir) {
        candidates.push(`${dir}/bash`)
      }
    }
  }

  for (const known of deps.knownLocations ?? ['/usr/bin/bash', '/bin/bash', '/usr/local/bin/bash']) {
    candidates.push(known)
  }

  // A candidate must be an executable file, not merely present: a
  // non-executable `bash` would fail at spawn time.
  const fileExists =
    deps.fileExists ??
    ((candidate: string) => {
      try {
        if (!statSync(candidate).isFile()) {
          return false
        }

        accessSync(candidate, constants.X_OK)

        return true
      } catch {
        return false
      }
    })

  return candidates.find(candidate => fileExists(candidate)) ?? null
}

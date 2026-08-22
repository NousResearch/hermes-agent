import fs from 'node:fs'
import path from 'node:path'

// The running app's .app bundle (packaged macOS): execPath is
// <App>.app/Contents/MacOS/<exe>; climb three levels to the bundle root.
//
// Only clearly transient launch locations are redirected to /Applications.
// An arbitrary alternate copy may be intentional (for example a beta or test
// build) and must keep updating/relaunching itself even when another install
// exists in /Applications.
function isTransientMacBundle(bundlePath: string): boolean {
  const normalized = path.normalize(bundlePath)
  const parts = normalized.split(path.sep)

  return normalized.startsWith(`/Volumes${path.sep}`) || parts.some((part) => part.startsWith('.hermes.preclone-'))
}

function canonicalMacBundle(bundlePath: string): string {
  return path.join('/Applications', path.basename(bundlePath))
}

export function runningAppBundle(
  execPath: string,
  existsSync: (p: string) => boolean = fs.existsSync,
  platform: NodeJS.Platform = process.platform
): string | null {
  if (platform !== 'darwin') {
    return null
  }

  let dir = path.dirname(execPath) // .../Contents/MacOS

  for (let i = 0; i < 2; i++) {
    dir = path.dirname(dir)
  } // -> .../X.app

  const running = dir.endsWith('.app') ? dir : null

  if (!running || running.startsWith(`/Applications${path.sep}`) || !isTransientMacBundle(running)) {
    return running
  }

  const canonical = canonicalMacBundle(running)

  return existsSync(canonical) ? canonical : running
}

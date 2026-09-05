import fs from 'node:fs'
import path from 'node:path'

/**
 * Classify whether a Desktop apply should use the runtime-free client-only
 * update path instead of `hermes update` / installer --repair.
 *
 * A present venv pair is always a full install. Remote/ssh/cloud with no
 * venv is an intentional thin client. Missing venv in local mode is a
 * broken install — do not paper over it with a client-only update.
 */

export interface ClientOnlyUpdateSurface {
  remoteMode: boolean
  hasVenvHermes: boolean
  hasVenvPython: boolean
}

export function isClientOnlyUpdateSurface(surface: ClientOnlyUpdateSurface): boolean {
  if (surface.hasVenvHermes && surface.hasVenvPython) {
    return false
  }

  return surface.remoteMode && !surface.hasVenvHermes && !surface.hasVenvPython
}

/** Read both runtime layouts supported by Desktop backend discovery. */
export function inspectClientOnlyUpdateSurface(
  installRoot: string,
  remoteMode: boolean,
  isWindows = process.platform === 'win32'
): ClientOnlyUpdateSurface {
  const executable = (file: string) => {
    try {
      fs.accessSync(file, fs.constants.X_OK)
      return fs.statSync(file).isFile()
    } catch { return false }
  }
  const bins = ['.venv', 'venv'].map(dir => path.join(installRoot, dir, isWindows ? 'Scripts' : 'bin'))
  return {
    remoteMode,
    hasVenvHermes: bins.some(bin => executable(path.join(bin, isWindows ? 'hermes.exe' : 'hermes'))),
    hasVenvPython: bins.some(bin => (isWindows ? ['python.exe'] : ['python3', 'python']).some(name => executable(path.join(bin, name))))
  }
}

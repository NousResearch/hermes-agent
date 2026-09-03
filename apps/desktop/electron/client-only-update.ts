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

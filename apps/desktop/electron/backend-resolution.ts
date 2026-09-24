/**
 * Precedence for Desktop backend resolution below the two explicit source
 * rungs (HERMES_DESKTOP_HERMES_ROOT and an unpackaged checkout).
 *
 * HERMES_DESKTOP_IGNORE_EXISTING=1 skips discovered rungs — the managed
 * install, `hermes` on PATH, and system Python — so no local serve starts
 * and the caller falls through to connect/onboarding. It does not skip an
 * explicit HERMES_DESKTOP_HERMES command.
 *
 * The post-bootstrap re-resolve must pass `justInstalled`. Otherwise the
 * flag would skip the runtime this process just installed and the installer
 * would see onboarding again and repeat.
 */
export type BackendRung =
  | 'explicit-root'
  | 'unpackaged-source'
  | 'active'
  | 'explicit-command'
  | 'path'
  | 'system-python'
  | 'onboarding'
  | 'installed-unusable'

export interface BackendResolutionInput {
  ignoreExisting: boolean
  justInstalled: boolean
  bootstrapRepairRequested: boolean
  hasExplicitRoot: boolean
  hasUnpackagedSource: boolean
  activeRuntimeUsable: boolean
  hasExplicitCommand: boolean
  hasPathHermes: boolean
  hasSystemPython: boolean
}

const NO_CANDIDATES: BackendResolutionInput = {
  ignoreExisting: false,
  justInstalled: false,
  bootstrapRepairRequested: false,
  hasExplicitRoot: false,
  hasUnpackagedSource: false,
  activeRuntimeUsable: false,
  hasExplicitCommand: false,
  hasPathHermes: false,
  hasSystemPython: false
}

export const INSTALLED_RUNTIME_UNUSABLE =
  'Hermes install finished, but the new runtime could not be resolved. Desktop will not start the installer again.'

export function isIgnoreExisting(value: string | undefined): boolean {
  return value === '1'
}

/** Options the re-resolve after a successful bootstrap must pass. */
export function postBootstrapResolveOptions(): { justInstalled: true } {
  return { justInstalled: true }
}

export function selectBackendRung(input: BackendResolutionInput): BackendRung {
  if (input.hasExplicitRoot) {
    return 'explicit-root'
  }

  if (input.hasUnpackagedSource) {
    return 'unpackaged-source'
  }

  const skipDiscovered = input.ignoreExisting && !input.justInstalled

  if (input.activeRuntimeUsable && !input.bootstrapRepairRequested && !skipDiscovered) {
    return 'active'
  }

  if (input.hasExplicitCommand) {
    return 'explicit-command'
  }

  if (!input.ignoreExisting && input.hasPathHermes) {
    return 'path'
  }

  if (!input.ignoreExisting && input.hasSystemPython) {
    return 'system-python'
  }

  // The installer already ran. Onboarding here would start it again; PATH and
  // system Python are the discovered runtimes the flag promised to skip.
  if (input.ignoreExisting && input.justInstalled) {
    return 'installed-unusable'
  }

  return 'onboarding'
}

export function shouldProbeActiveRuntime(input: {
  ignoreExisting: boolean
  justInstalled: boolean
  bootstrapRepairRequested: boolean
}): boolean {
  return (
    selectBackendRung({
      ...NO_CANDIDATES,
      ...input,
      activeRuntimeUsable: true
    }) === 'active'
  )
}

export function shouldProbeDiscoveredPath(input: { ignoreExisting: boolean }): boolean {
  return (
    selectBackendRung({
      ...NO_CANDIDATES,
      ignoreExisting: input.ignoreExisting,
      hasPathHermes: true
    }) === 'path'
  )
}

export function shouldProbeSystemPython(input: { ignoreExisting: boolean }): boolean {
  return (
    selectBackendRung({
      ...NO_CANDIDATES,
      ignoreExisting: input.ignoreExisting,
      hasSystemPython: true
    }) === 'system-python'
  )
}

export function unresolvedDiscoveredRuntime(input: {
  ignoreExisting: boolean
  justInstalled: boolean
}): 'onboarding' | 'installed-unusable' {
  const rung = selectBackendRung({
    ...NO_CANDIDATES,
    ignoreExisting: input.ignoreExisting,
    justInstalled: input.justInstalled,
    hasPathHermes: true,
    hasSystemPython: true
  })

  return rung === 'installed-unusable' ? 'installed-unusable' : 'onboarding'
}

export const IGNORE_EXISTING_ENV = 'HERMES_DESKTOP_IGNORE_EXISTING'

/**
 * Returns true if Desktop was asked to skip discovered local runtimes.
 *
 * Explicit developer/deployment overrides (HERMES_DESKTOP_HERMES_ROOT,
 * development checkout, HERMES_DESKTOP_HERMES) are operator choices and are
 * NOT skipped by this flag. Everything below them on the resolve ladder — the
 * managed install at ACTIVE_HERMES_ROOT, `hermes` on PATH, the system-python
 * hermes_cli module — is *discovered*, and that is what "ignore existing"
 * promises to skip.
 */
export function shouldIgnoreDiscoveredLocalRuntimes(env: NodeJS.ProcessEnv = process.env): boolean {
  return env[IGNORE_EXISTING_ENV] === '1'
}

export interface ActiveBackendResolutionOptions {
  activeRuntimeUsable?: boolean
  bootstrapRepairRequested: boolean
  ignoreExisting?: boolean
  env?: NodeJS.ProcessEnv
}

/** Whether the managed runtime should even be probed/evaluated. */
export function canActiveBackendResolve({
  bootstrapRepairRequested,
  ignoreExisting,
  env
}: Omit<ActiveBackendResolutionOptions, 'activeRuntimeUsable'>): boolean {
  const isIgnored = ignoreExisting ?? shouldIgnoreDiscoveredLocalRuntimes(env)

  return !bootstrapRepairRequested && !isIgnored
}

/** Whether the usable managed runtime may win backend resolution. */
export function shouldUseActiveBackend({
  activeRuntimeUsable = false,
  bootstrapRepairRequested,
  ignoreExisting,
  env
}: ActiveBackendResolutionOptions): boolean {
  if (!canActiveBackendResolve({ bootstrapRepairRequested, ignoreExisting, env })) {
    return false
  }

  return Boolean(activeRuntimeUsable)
}

export type SystemPythonResolutionInput =
  | NodeJS.ProcessEnv
  | boolean
  | { ignoreExisting?: boolean; env?: NodeJS.ProcessEnv }

/** Whether the system-Python fallback may win backend resolution. */
export function shouldUseSystemPythonBackend(
  optionsOrEnv: SystemPythonResolutionInput = process.env
): boolean {
  if (typeof optionsOrEnv === 'boolean') {
    return !optionsOrEnv
  }
  if (optionsOrEnv && typeof optionsOrEnv === 'object') {
    if ('ignoreExisting' in optionsOrEnv || 'env' in optionsOrEnv) {
      const opts = optionsOrEnv as { ignoreExisting?: boolean; env?: NodeJS.ProcessEnv }
      const isIgnored = opts.ignoreExisting ?? shouldIgnoreDiscoveredLocalRuntimes(opts.env)

      return !isIgnored
    }

    return !shouldIgnoreDiscoveredLocalRuntimes(optionsOrEnv as NodeJS.ProcessEnv)
  }

  return !shouldIgnoreDiscoveredLocalRuntimes()
}

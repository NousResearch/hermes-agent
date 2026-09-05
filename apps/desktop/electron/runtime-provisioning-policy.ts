/**
 * The Desktop resolves local runtimes for two materially different purposes:
 *
 * - primary-bootstrap: the user is starting the primary local Desktop backend,
 *   so first-run provisioning is allowed;
 * - secondary-activate: a background profile or registry source may reuse an
 *   installed runtime, but must never turn observation/UI routing into an
 *   implicit platform install.
 *
 * Keeping this policy inside ensureRuntime() means every current and future
 * caller must declare its authority before the bootstrap sentinel can reach the
 * installer. A caller-local guard can fix one path while the next secondary
 * path reopens the same deployment-topology bug.
 */

export type RuntimeProvisioningIntent = 'primary-bootstrap' | 'secondary-activate'

export interface ResolvedRuntimeLike {
  bootstrap?: unknown
  kind?: unknown
}

export const SECONDARY_RUNTIME_NOT_INSTALLED =
  'No local Hermes runtime is installed on this machine. Install Hermes locally before using "This device" or a secondary local profile.'

export class RuntimeProvisioningDeniedError extends Error {
  readonly code = 'runtime-provisioning-not-allowed'
  readonly intent: RuntimeProvisioningIntent

  constructor(intent: RuntimeProvisioningIntent) {
    super(SECONDARY_RUNTIME_NOT_INSTALLED)
    this.name = 'RuntimeProvisioningDeniedError'
    this.intent = intent
  }
}

export function runtimeNeedsProvisioning(runtime: null | ResolvedRuntimeLike | undefined): boolean {
  return Boolean(runtime && typeof runtime === 'object' && runtime.kind === 'bootstrap-needed')
}

export function assertRuntimeProvisioningAllowed(
  runtime: null | ResolvedRuntimeLike | undefined,
  intent: RuntimeProvisioningIntent
): void {
  if (runtimeNeedsProvisioning(runtime) && intent !== 'primary-bootstrap') {
    throw new RuntimeProvisioningDeniedError(intent)
  }
}

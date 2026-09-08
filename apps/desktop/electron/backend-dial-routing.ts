import type { BackendDialClaims } from './backend-dial-claim'

export interface BackendDialRoutingDeps {
  claims: Pick<BackendDialClaims, 'run'>
  scopeKey: (connectionId: string | null, profile: string | null | undefined) => string
}

/** Route one backend bootstrap through the main-process single-owner claim. */
export function runBackendDial<T>(
  deps: BackendDialRoutingDeps,
  connectionId: string | null,
  profile: string | null | undefined,
  dial: () => Promise<T> | T,
  claimKey?: string
): Promise<T> {
  return deps.claims.run(claimKey ?? deps.scopeKey(connectionId, profile), dial)
}

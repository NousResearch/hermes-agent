import type { BackendDialClaims } from './backend-dial-claim'
import {
  connectionScopeKey,
  pathForRegistryBackendRequest,
  type ProfileRouteOptions,
  type RegistryBackendRequestScope,
  resolveProfileApiRequest
} from './connection-config'
import { backendScopeKey, LOCAL_CONNECTION_ID } from './connection-registry'
import { DEFAULT_FETCH_TIMEOUT_MS, resolveTimeoutMs } from './hardening'
import { tagRegistrySessionResponse } from './profile-session-routing'

export interface RegistryApiRequest {
  path: string
  profile?: null | string
  passive?: boolean
  method?: string
  body?: unknown
  upload?: unknown
  timeoutMs?: number
}

interface RegistryApiBackend extends RegistryBackendRequestScope {
  baseUrl: string
}

interface RegistryApiDispatchDeps {
  backendDialClaims: BackendDialClaims
  ensureRegistryBackend: (
    connectionId: string,
    profile?: null | string,
    managedUpdateCorrelation?: string,
    options?: { passive?: boolean }
  ) => Promise<RegistryApiBackend>
  fetchJsonForBackend: (
    connection: RegistryApiBackend,
    path: string,
    options: Pick<RegistryApiRequest, 'method' | 'body' | 'upload' | 'timeoutMs'>
  ) => Promise<unknown>
  profileRouteOptions: (profile: null | string | undefined, request: RegistryApiRequest) => ProfileRouteOptions
}

export function createRegistryApiDispatcher(deps: RegistryApiDispatchDeps) {
  return async function dispatchRegistryApiRequest(
    request: RegistryApiRequest,
    registryConnectionId: string,
    routeProfile = request?.profile,
    requestProfile = request?.profile
  ): Promise<unknown> {
    let backendProfile = routeProfile
    let path = request.path

    if (registryConnectionId === LOCAL_CONNECTION_ID) {
      const options = deps.profileRouteOptions(routeProfile, request)

      // Only a proven-local primary may serve another local profile. Remote
      // primaries and per-profile overrides retain the registry resolver's
      // isolation; the existing REST policy owns the method/path allowlist.
      if (!options.globalRemote && !options.primaryRemoteActive && !options.profileRemoteOverride) {
        const route = resolveProfileApiRequest(routeProfile, path, options)

        if (route.backendProfile === null) {
          backendProfile = connectionScopeKey(options.primaryProfile) || 'default'
          // Deletion may route via the primary while naming a different
          // target. Backend selection must not replace that request scope.
          path = resolveProfileApiRequest(requestProfile, path, options).requestPath
        }
      }
    }

    // Passive reads stay outside the claim: an interactive dial must not
    // coalesce onto their "no warm backend" rejection.
    const connection = request?.passive
      ? await deps.ensureRegistryBackend(registryConnectionId, backendProfile, '', { passive: true })
      : await deps.backendDialClaims.run(backendScopeKey(registryConnectionId, backendProfile), () =>
          deps.ensureRegistryBackend(registryConnectionId, backendProfile)
        )

    const requestPath = pathForRegistryBackendRequest(path, requestProfile, connection)

    const response = await deps.fetchJsonForBackend(connection, requestPath, {
      method: request?.method,
      body: request?.body,
      upload: request?.upload,
      timeoutMs: resolveTimeoutMs(request?.timeoutMs, DEFAULT_FETCH_TIMEOUT_MS)
    })

    return (request?.method || 'GET').toUpperCase() === 'GET'
      ? tagRegistrySessionResponse(requestPath, response, registryConnectionId)
      : response
  }
}

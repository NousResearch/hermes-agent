import { backendScopeKey, type RegistryLocalRoute } from './connection-registry'

interface RegistryApiRequest {
  path: string
  method?: string
  body?: unknown
  upload?: unknown
  profile?: unknown
  passive?: boolean
  timeoutMs?: number
}

/** The same backend selection and REST dispatch used by main's hermes:api IPC. */
export async function dispatchRegistryApiRequest<Backend extends { mode: string }, Response>(
  request: RegistryApiRequest,
  connectionId: string,
  routeProfile: string | undefined,
  requestProfile: string | undefined,
  deps: {
    afterProfileRequest: (connectionId: string, request: RegistryApiRequest, response: Response, mode: string) => void
    claim: (key: string, ensure: () => Promise<Backend>) => Promise<Backend>
    ensure: (connectionId: string, profile: string | undefined, passive: boolean) => Promise<Backend>
    fetch: (
      backend: Backend,
      path: string,
      options: { method?: string; body?: unknown; upload?: unknown; timeoutMs: number }
    ) => Promise<Response>
    localRoute: (profile: string | undefined) => RegistryLocalRoute
    path: (path: string, profile: string | undefined, backend: Backend, method: string) => string
    timeout: (timeoutMs: number | undefined) => number
    tag: (path: string, response: Response, connectionId: string) => Response
  }
): Promise<Response> {
  // Passive reads stay OUT of the claim: an interactive open must never
  // coalesce onto their no-warm-backend rejection.
  // Persisted owner hints can route a background GET to This device long
  // after the primary moved remote. Always resolve those GETs passively:
  // sampling a warm pool here races with eviction before ensure. A caller
  // preparing an explicit write opts into a dial with passive: false.
  const localRoute =
    connectionId === 'local' && (request.method || 'GET').toUpperCase() === 'GET' ? deps.localRoute(routeProfile) : null

  const passive =
    request.passive === true ||
    Boolean(localRoute && !localRoute.delegate && !localRoute.refuse && request.passive !== false)

  const backend = passive
    ? await deps.ensure(connectionId, routeProfile, true)
    : await deps.claim(backendScopeKey(connectionId, routeProfile), () =>
        deps.ensure(connectionId, routeProfile, false)
      )

  const requestPath = deps.path(request.path, requestProfile, backend, request.method || 'GET')

  const response = await deps.fetch(backend, requestPath, {
    method: request.method,
    body: request.body,
    upload: request.upload,
    timeoutMs: deps.timeout(request.timeoutMs)
  })

  deps.afterProfileRequest(connectionId, request, response, backend.mode)

  return (request.method || 'GET').toUpperCase() === 'GET' ? deps.tag(requestPath, response, connectionId) : response
}

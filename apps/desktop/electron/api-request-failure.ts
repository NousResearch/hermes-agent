export interface ApiRequest {
  method?: string
  path?: string
}

/**
 * Expected 404 probes are quiet, except mutations: a failed delete must retain
 * its request path in desktop.log so the affected session can be identified.
 */
export function shouldLogExpectedNotFoundApiFailure(request: ApiRequest | null | undefined): boolean {
  return String(request?.method ?? 'GET').toUpperCase() === 'DELETE'
}

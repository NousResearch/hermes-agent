/**
 * Profiles the primary backend's gateway serves from the shared home.
 *
 * Under `gateway.multiplex_profiles` one gateway process serves every profile
 * and reports them as `gateway_shared_with` on `/api/status`. The renderer
 * already polls that route, so main records the field from the status response
 * it forwards rather than fetching the route a second time for routing.
 *
 * Process-local and never persisted: a restart of the primary backend clears
 * the record until the next status poll, which errs toward giving the profile
 * its own backend instead of assuming a shared home.
 */
let sharedProfiles: null | string[] = null

export function recordGatewaySharedProfiles(value: unknown): void {
  sharedProfiles = Array.isArray(value) ? value.filter(entry => typeof entry === 'string') : null
}

export function gatewaySharedProfiles(): null | string[] {
  return sharedProfiles
}

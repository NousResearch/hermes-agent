/**
 * A gateway scope becomes authoritative only after its socket is open.
 *
 * The renderer mounts before the managed backend is ready. Recording the
 * initial scope while it is still connecting suppresses the first successful
 * refresh and leaves the composer's persisted model from the previous run in
 * charge of boot state.
 */
export function shouldRefreshGatewayScope(lastScope: null | string, scope: string, gatewayState: string): boolean {
  return gatewayState === 'open' && lastScope !== scope
}
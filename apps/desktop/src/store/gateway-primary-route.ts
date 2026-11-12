import { atom } from 'nanostores'

/** The window primary socket's own route, whichever gateway is in front. */
export interface PrimaryGatewayRoute {
  /** Registry source the primary socket serves, when known. */
  connectionId: null | string
  /** Resolved mode of the primary's descriptor: a `local` primary is ONE
   *  `hermes serve --profile <primary>` child and can never stand in for a
   *  pooled profile's own backend. */
  mode: 'local' | 'remote' | null
}

const ROUTE_KEY = Symbol.for('hermes.desktop.primaryGatewayRoute')

// Same dev-HMR rule as the gateway registry state: a re-evaluated module must
// hand back the SAME atom, or existing subscribers are orphaned. Production has
// no `import.meta.hot`, so this is a plain module-local atom there.
function routeAtom() {
  const initial: PrimaryGatewayRoute = { connectionId: null, mode: null }

  if (import.meta.hot) {
    const store = globalThis as unknown as { [ROUTE_KEY]?: ReturnType<typeof atom<PrimaryGatewayRoute>> }

    return (store[ROUTE_KEY] ??= atom<PrimaryGatewayRoute>(initial))
  }

  return atom<PrimaryGatewayRoute>(initial)
}

/** store/gateway.ts's setPrimaryGateway / setPrimaryGatewayConnectionId are its only writers.
 *  Unlike $connection, which follows the gateway in front (a foregrounded local secondary
 *  reports `local` beside a remote primary) and goes null on a reconnect blip, this keeps
 *  the last recorded primary until a new primary socket replaces it. */
export const $primaryGatewayRoute = routeAtom()

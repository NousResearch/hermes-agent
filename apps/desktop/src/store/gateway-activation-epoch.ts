import { useStore } from '@nanostores/react'
import { atom } from 'nanostores'

const EPOCH_KEY = Symbol.for('hermes.desktop.gatewayActivationEpoch')

// Same dev-HMR rule as the gateway registry state: a re-evaluated module must
// hand back the SAME atom, or existing subscribers are orphaned. Production has
// no `import.meta.hot`, so this is a plain module-local atom there.
function epochAtom() {
  if (import.meta.hot) {
    const store = globalThis as unknown as { [EPOCH_KEY]?: ReturnType<typeof atom<number>> }

    return (store[EPOCH_KEY] ??= atom<number>(0))
  }

  return atom<number>(0)
}

/** The gateway activation epoch: bumped on every active route selection,
 *  including same-profile source swaps. store/gateway.ts's
 *  beginGatewayActivation is its only writer; read it imperatively with
 *  `gatewayActivationEpoch()`. */
export const $gatewayActivationEpoch = epochAtom()

/** `gatewayActivationEpoch()` for a component. A render that keys on the epoch
 *  MUST read it here: a plain getter call is a read nothing subscribes to, so a
 *  memoized render (the React Compiler) keeps the epoch it first saw while an
 *  activation moves it underneath without touching connection/profile/gateway. */
export function useGatewayActivationEpoch(): number {
  return useStore($gatewayActivationEpoch)
}

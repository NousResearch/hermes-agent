import {
  closeSecondaryGateways, configureGatewayRegistry, openGatewayForAgent,
  openGatewayForProfile, openSecondaryCount, parkSecondariesForRetiredBackend,
  reconnectSecondaryGateways,
} from '../store/gateway'

const parked: Record<string, string[]> = {}
const scopes = new Set(['idle-a', 'conn:local::idle-a', 'idle-b', 'conn:local::idle-b'])
configureGatewayRegistry({ onEvent: () => {}, foregroundScopes: () => scopes })

interface FixtureWindow extends Window {
  retirementFixture: { onRetiring: (handler: (key: string) => void) => void }
}
;(window as unknown as FixtureWindow).retirementFixture.onRetiring(key => {
  parked[key] = parkSecondariesForRetiredBackend(key)
})

const api = {
  async open() {
    for (const key of ['idle-a', 'idle-b']) {
      await openGatewayForProfile(key)
      await openGatewayForAgent('local', key)
    }

    return openSecondaryCount()
  },
  async wake() {
    reconnectSecondaryGateways()
    reconnectSecondaryGateways({ forceOpenSockets: true })
    // Observe past the normal first reconnect timer, not only its microtask.
    await new Promise(resolve => setTimeout(resolve, 2500))

    return { parked, open: openSecondaryCount() }
  },
  close: closeSecondaryGateways,
}

Object.assign(window, { poolRetirementRenderer: api })

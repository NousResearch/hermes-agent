import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import type { HermesGateway } from '@/hermes'

import {
  $gateway,
  activeGateway,
  closeSecondaryGateways,
  ensureGatewayForProfile,
  setPrimaryGateway
} from './gateway'

describe('activeGateway profile isolation', () => {
  beforeEach(async () => {
    closeSecondaryGateways()
    setPrimaryGateway(null)
    await ensureGatewayForProfile('default')
  })

  afterEach(async () => {
    closeSecondaryGateways()
    setPrimaryGateway(null)
    await ensureGatewayForProfile('default')
  })

  it('does not substitute a newly-primary gateway for the previously active profile', () => {
    const primary = { connectionState: 'open' } as HermesGateway

    // The primary backend can be re-homed while the store still has the
    // previous profile as active. Until that profile's own socket is active,
    // callers must see no gateway rather than send to the new primary.
    setPrimaryGateway(primary, 'work')

    expect(activeGateway()).toBeNull()
    expect($gateway.get()).toBeNull()
  })
})

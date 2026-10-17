import { afterEach, describe, expect, it } from 'vitest'

import { $busyInputConfig, busyInputOwnerKey, configuredBusyInputMode } from '@/store/busy-input-mode'

import { busyModeHasCarrier, normalizeBusyInputMode } from './busy-input-mode'

afterEach(() => $busyInputConfig.set(null))

describe('display.busy_input_mode config scope', () => {
  const connection = { connectionId: 'gateway-a' }
  const profile = 'default'

  it('reflects the active backend owner without a second global atom', () => {
    $busyInputConfig.set({ owner: busyInputOwnerKey(connection.connectionId, profile), connection, mode: 'steer' })
    expect(configuredBusyInputMode($busyInputConfig.get(), connection, profile)).toBe('steer')
    expect(configuredBusyInputMode($busyInputConfig.get(), connection, 'other')).toBe('interrupt')
    expect(configuredBusyInputMode($busyInputConfig.get(), { connectionId: 'gateway-b' }, profile)).toBe('interrupt')
  })

  it('tracks config changes and shares the backend normalizer', () => {
    $busyInputConfig.set({ owner: busyInputOwnerKey(connection.connectionId, profile), connection, mode: 'queue' })
    expect(configuredBusyInputMode($busyInputConfig.get(), connection, profile)).toBe('queue')
    $busyInputConfig.set({ owner: busyInputOwnerKey(connection.connectionId, profile), connection, mode: 'steer' })
    expect(configuredBusyInputMode($busyInputConfig.get(), connection, profile)).toBe('steer')
    expect(normalizeBusyInputMode(' Stre ')).toBe('interrupt')
  })
})

// The other seam: the composer's capability check and the submit path must ask
// for the SAME carrier. `interrupt` rides onSteer (session.redirect) and
// `steer` rides onSteerHidden (session.steer); a check that consulted the wrong
// one would let the UI advertise an action the submit path then cannot deliver.
describe('busy-input carrier selection', () => {
  const both = { onSteer: true, onSteerHidden: true }

  it('asks for onSteerHidden in steer mode', () => {
    expect(busyModeHasCarrier('steer', { onSteer: true, onSteerHidden: true })).toBe(true)
    // onSteer alone cannot deliver a steer — the UI must not promise one.
    expect(busyModeHasCarrier('steer', { onSteer: true, onSteerHidden: false })).toBe(false)
  })

  it('asks for onSteer in interrupt mode', () => {
    expect(busyModeHasCarrier('interrupt', { onSteer: true, onSteerHidden: true })).toBe(true)
    // onSteerHidden alone cannot deliver a redirect.
    expect(busyModeHasCarrier('interrupt', { onSteer: false, onSteerHidden: true })).toBe(false)
  })

  it('ignores carriers in queue mode, which needs neither RPC', () => {
    // resolveBusyComposerAction returns for queue mode before it ever reads
    // canCorrect, so the carrier question is moot there — the value must simply
    // not invent a dependency on a callback queue mode never calls.
    expect(() => busyModeHasCarrier('queue', both)).not.toThrow()
    expect(busyModeHasCarrier('queue', { onSteer: false, onSteerHidden: false })).toBe(false)
  })
})

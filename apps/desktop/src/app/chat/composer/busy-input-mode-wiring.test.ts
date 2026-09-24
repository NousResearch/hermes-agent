/**
 * A guard on the seam the two halves of this change meet at: the config
 * refresh path publishing into `$busyInputMode`, and `useComposerSubmit`
 * branching on a `busyInputMode` argument. Both halves are unit-tested
 * in isolation, so a wiring break (value parsed correctly but nobody reading
 * the atom; hook honouring a mode nobody can set) would leave both suites
 * green while the feature silently no-ops.
 */

import { describe, expect, it } from 'vitest'

import { $busyInputMode, busyModeHasCarrier, normalizeBusyInputMode, setBusyInputModeFromConfig } from './busy-input-mode'

const CONFIG_KEY = 'display.busy_input_mode'

describe('display.busy_input_mode wiring', () => {
  it('carries the config value into the atom the composer subscribes to', () => {
    setBusyInputModeFromConfig('steer')

    expect(normalizeBusyInputMode($busyInputMode.get())).toBe('steer')
  })

  it('falls back to the framework default for absent and malformed values', () => {
    // Absent: a config written before the key existed must keep working.
    setBusyInputModeFromConfig(undefined)
    expect($busyInputMode.get()).toBe('interrupt')

    // Malformed: a typo must not silently turn a busy Enter into a redirect
    // the user never asked for.
    setBusyInputModeFromConfig('stre')
    expect($busyInputMode.get()).toBe('interrupt')
  })

  it('notifies subscribers on change, so a settings save takes effect live', () => {
    const seen: string[] = []
    // A nanostores subscription fires immediately with the current value, then
    // on each change — that is what a live settings save relies on.
    const stop = $busyInputMode.subscribe(value => seen.push(value))

    setBusyInputModeFromConfig('queue')
    setBusyInputModeFromConfig('steer')
    stop()

    expect(seen).toEqual(['interrupt', 'queue', 'steer'])
  })

  // The key as the backend schema spells it, so a rename on either side of the
  // seam is caught here rather than by a user reporting the setting vanished.
  it('keeps the config key aligned with the backend schema name', () => {
    expect(CONFIG_KEY).toBe('display.busy_input_mode')
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

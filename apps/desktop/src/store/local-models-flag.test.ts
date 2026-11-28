import { describe, expect, it } from 'vitest'

import { $localModelsEnabled, localModelsEnabled } from './local-models-flag'

// Desktop: the Electron preload bridge is the sole source of truth (`--local`
// launch flag). Webapp (browser): no bridge — opt in with `?local=1` so the
// local-models surfaces (e.g. the multi-GPU System resources chip) read the
// backend machine's local models.
describe('localModelsEnabled', () => {
  it('desktop bridge on: enabled', () => {
    expect(localModelsEnabled({ localModelsEnabled: true }, '')).toBe(true)
  })
  it('desktop bridge present but flag off: disabled, query param ignored', () => {
    expect(localModelsEnabled({ localModelsEnabled: false }, '?local=1')).toBe(false)
  })
  it('desktop bridge without the field: disabled', () => {
    expect(localModelsEnabled({}, '?local=1')).toBe(false)
  })
  it('webapp (no bridge): ?local=1 opts in', () => {
    expect(localModelsEnabled(undefined, '?profile=hermes-ops&local=1')).toBe(true)
  })
  it('webapp (no bridge): absent or other values stay off', () => {
    expect(localModelsEnabled(undefined, '')).toBe(false)
    expect(localModelsEnabled(undefined, '?profile=hermes-ops')).toBe(false)
    expect(localModelsEnabled(undefined, '?local=0')).toBe(false)
    expect(localModelsEnabled(undefined, '?local=true')).toBe(false)
  })
  it('initialises from the window globals (test env: no bridge, empty search)', () => {
    expect($localModelsEnabled.get()).toBe(false)
  })
})

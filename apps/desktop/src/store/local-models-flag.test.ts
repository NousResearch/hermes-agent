import { beforeEach, describe, expect, it, vi } from 'vitest'

describe('$localModelsEnabled', () => {
  beforeEach(() => {
    vi.resetModules()
  })

  it('reads true when the preload bridge reports the --local launch flag', async () => {
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { localModelsEnabled: true }
    })

    const { $localModelsEnabled } = await import('./local-models-flag')

    expect($localModelsEnabled.get()).toBe(true)
  })

  it('defaults to false when the bridge omits the flag (older preload, web)', async () => {
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {}
    })

    const { $localModelsEnabled } = await import('./local-models-flag')

    expect($localModelsEnabled.get()).toBe(false)
  })

  it('defaults to false with no bridge at all', async () => {
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: undefined
    })

    const { $localModelsEnabled } = await import('./local-models-flag')

    expect($localModelsEnabled.get()).toBe(false)
  })
})

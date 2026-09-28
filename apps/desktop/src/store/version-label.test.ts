import { afterEach, describe, expect, it } from 'vitest'

import { $versionLabel, setVersionLabelFromConfig } from './version-label'

afterEach(() => {
  setVersionLabelFromConfig(undefined)
})

describe('version label config bridge', () => {
  it('defaults to the distance form and adopts only the release opt-in', () => {
    expect($versionLabel.get()).toBe('release+distance')

    setVersionLabelFromConfig('release')
    expect($versionLabel.get()).toBe('release')

    // Unknown or absent values fall back to the historical default.
    setVersionLabelFromConfig('release+distance')
    expect($versionLabel.get()).toBe('release+distance')

    setVersionLabelFromConfig(undefined)
    expect($versionLabel.get()).toBe('release+distance')

    setVersionLabelFromConfig('nonsense')
    expect($versionLabel.get()).toBe('release+distance')
  })
})

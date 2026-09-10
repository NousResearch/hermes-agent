import { describe, expect, it } from 'vitest'

import { LUNAR_CITY_ASSET_MANIFEST } from './world-assets'

describe('Lunar City asset manifest', () => {
  it('covers the baseline terrain, road network, buildings, and character roles', () => {
    expect(LUNAR_CITY_ASSET_MANIFEST.schemaVersion).toBe(1)
    expect(LUNAR_CITY_ASSET_MANIFEST.profileManifest).toBe('lunar-city/profile-assets.json')
    expect(LUNAR_CITY_ASSET_MANIFEST.assets.filter(asset => asset.kind === 'building')).toHaveLength(8)
    expect(
      LUNAR_CITY_ASSET_MANIFEST.assets.filter(asset => asset.kind === 'character').map(asset => asset.role)
    ).toEqual(['leader', 'worker', 'dispatcher'])
    expect(LUNAR_CITY_ASSET_MANIFEST.assets.some(asset => asset.id === 'terrain-colony-basin')).toBe(true)
    expect(LUNAR_CITY_ASSET_MANIFEST.assets.some(asset => asset.id === 'road-network-primary')).toBe(true)
  })
})

export type WorldAssetKind = 'building' | 'character' | 'road' | 'terrain'

export interface WorldAssetManifestEntry {
  collection: 'Buildings' | 'Characters' | 'Roads' | 'Terrain'
  id: string
  kind: WorldAssetKind
  lod: ('high' | 'low')[]
  role?: string
}

export interface WorldAssetManifest {
  assets: WorldAssetManifestEntry[]
  profileManifest: string
  renderProfile: string
  scene: string
  schemaVersion: 1
  source: string
  units: 'meters'
}

export const LUNAR_CITY_ASSET_MANIFEST: WorldAssetManifest = {
  assets: [
    { collection: 'Terrain', id: 'terrain-colony-basin', kind: 'terrain', lod: ['high', 'low'] },
    { collection: 'Roads', id: 'road-network-primary', kind: 'road', lod: ['high', 'low'] },
    ...[
      ['library', 'knowledge'],
      ['research-lab', 'research'],
      ['arts-studio', 'creative'],
      ['council-hall', 'governance'],
      ['engineering-workshop', 'engineering'],
      ['triage-clinic', 'medical'],
      ['review-office', 'review'],
      ['archive', 'archive']
    ].map(([id, role]) => ({
      collection: 'Buildings' as const,
      id,
      kind: 'building' as const,
      lod: ['high', 'low'] as ('high' | 'low')[],
      role
    })),
    { collection: 'Characters', id: 'leader-prototype', kind: 'character', lod: ['high', 'low'], role: 'leader' },
    { collection: 'Characters', id: 'worker-prototype', kind: 'character', lod: ['high', 'low'], role: 'worker' },
    { collection: 'Characters', id: 'dispatcher-cube', kind: 'character', lod: ['high', 'low'], role: 'dispatcher' }
  ],
  renderProfile: 'desktop-preview',
  profileManifest: 'lunar-city/profile-assets.json',
  scene: 'lunar-city-baseline',
  schemaVersion: 1,
  source: 'procedural-blender-baseline',
  units: 'meters'
}

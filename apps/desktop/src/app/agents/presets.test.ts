import { describe, expect, it } from 'vitest'

import { availablePresets, initialPresetMetadata, normalizePresetMetadata, skillAvailability } from './presets'

describe('subagent preset catalog', () => {
  it('seeds named presets with backend skill identities', () => {
    const metadata = initialPresetMetadata('2026-09-26T10:00:00.000Z')
    expect(metadata.schema_version).toBe(1)
    expect(new Set(metadata.presets.map(preset => preset.id)).size).toBe(metadata.presets.length)
    expect(metadata.presets.every(preset => preset.skills.every(skill => !skill.name.includes('/')))).toBe(true)
  })

  it('offers templates without replacing a saved role or duplicating its identity', () => {
    const saved = { ...initialPresetMetadata().presets[0], character: 'My own instructions' }
    const presets = availablePresets([saved])
    expect(presets.filter(preset => preset.id === saved.id)).toEqual([saved])
    expect(presets.length).toBeGreaterThan(1)
  })

  it('maps availability from the selected backend, including missing skills', () => {
    expect(skillAvailability([{ name: 'installed' }, { name: 'disabled' }, { name: 'remote-only' }], [
      { name: 'installed', enabled: true },
      { name: 'disabled', enabled: false }
    ])).toEqual([
      { name: 'installed', state: 'enabled' },
      { name: 'disabled', state: 'disabled' },
      { name: 'remote-only', state: 'missing' }
    ])
  })

  it('rejects malformed metadata without inventing a persisted preset', () => {
    expect(normalizePresetMetadata({ schema_version: 2, presets: [{ id: 'bad' }] })).toEqual({ schema_version: 1, presets: [] })
  })
})

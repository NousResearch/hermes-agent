import { describe, expect, it } from 'vitest'

import { CONFIG_SUBPAGES, configSubpageForField } from './config-subpages'
import { sectionFieldEntries } from './helpers'
import { movedSettingsTabRedirect } from './moved-tabs'

describe('Subagents settings destination', () => {
  it('moves every delegation field, including config-present fields, into Model only', () => {
    const config = {
      delegation: {
        model: 'm',
        provider: 'p',
        max_iterations: 9,
        base_url: 'https://example.invalid',
        request_overrides: { temperature: 0.2 }
      }
    }

    const entries = sectionFieldEntries({}, config)
    expect(entries.get('model')?.map(([key]) => key)).toEqual(
      expect.arrayContaining(['delegation.model', 'delegation.provider', 'delegation.max_iterations'])
    )
    expect(entries.get('advanced')?.filter(([key]) => key.startsWith('delegation.')) ?? []).toEqual([])
    expect(CONFIG_SUBPAGES.model.map(p => p.id)).toContain('delegation')
    expect(CONFIG_SUBPAGES.advanced.map(p => p.id)).not.toContain('delegation')
    expect(configSubpageForField('model', 'delegation.request_overrides.temperature')).toBe('delegation')
  })
  it('keeps old Advanced bookmarks and field links working without losing scope', () => {
    const result = movedSettingsTabRedirect('?tab=config%3Aadvanced&page=delegation&profile=B&field=delegation.model')
    expect(result).not.toBeNull()
    const params = new URLSearchParams(result!.split('?')[1])
    expect(params.get('tab')).toBe('config:model')
    expect(params.get('page')).toBe('delegation')
    expect(params.get('profile')).toBe('B')
    expect(params.get('field')).toBe('delegation.model')
  })
})

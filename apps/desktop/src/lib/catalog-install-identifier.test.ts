import { skillCatalogInstallIdentifier, skillCatalogInstallUrl } from '@hermes/shared'
import { describe, expect, it } from 'vitest'

describe('catalog skill install identifiers', () => {
  it('does not turn a missing community identifier into an ambiguous short name', () => {
    const skill = { name: 'humanizer', source: 'ClawHub' }

    expect(skillCatalogInstallIdentifier(skill)).toBeNull()
    expect(skillCatalogInstallUrl(skill)).toBeNull()
  })

  it('keeps the exact source-qualified ClawHub owner identifier', () => {
    const skill = { name: 'humanizer', source: 'ClawHub', identifier: '@owner/humanizer' }

    expect(skillCatalogInstallIdentifier(skill)).toBe('clawhub/@owner/humanizer')
  })

  it('preserves the optional catalog fallback', () => {
    expect(skillCatalogInstallIdentifier({ name: 'pdf', source: 'optional' })).toBe('official/pdf')
  })
})

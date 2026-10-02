import { describe, expect, it } from 'vitest'

import { buildSettingsPageSearch } from './subpages'

describe('openSettingsPage navigation', () => {
  it('lands a parent row on the view top-level page (no page param)', () => {
    const search = buildSettingsPageSearch('', 'config:appearance')
    const params = new URLSearchParams(search)

    expect(params.get('tab')).toBe('config:appearance')
    expect(params.get('page')).toBeNull()
  })

  it('keeps explicit child navigation deep-linking with ?page=', () => {
    const search = buildSettingsPageSearch('', 'config:appearance', 'general')
    const params = new URLSearchParams(search)

    expect(params.get('tab')).toBe('config:appearance')
    expect(params.get('page')).toBe('general')
  })

  it('clears stale subpage state while keeping unrelated params', () => {
    const search = buildSettingsPageSearch(
      '?tab=config%3Amodel&page=theme&field=theme&setting=x&foo=bar',
      'config:appearance'
    )

    const params = new URLSearchParams(search)

    expect(params.get('tab')).toBe('config:appearance')
    expect(params.get('page')).toBeNull()
    expect(params.get('field')).toBeNull()
    expect(params.get('setting')).toBeNull()
    expect(params.get('foo')).toBe('bar')
  })
})

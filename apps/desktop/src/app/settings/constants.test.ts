import { describe, expect, it } from 'vitest'

import { ENUM_OPTIONS, SECTIONS } from './constants'

describe('Power settings surface', () => {
  it('exposes every prevent-sleep field in a dedicated section', () => {
    const power = SECTIONS.find(section => section.id === 'power')

    expect(power?.keys).toEqual([
      'power.prevent_sleep.enabled',
      'power.prevent_sleep.surfaces',
      'power.prevent_sleep.mode'
    ])
  })

  it('limits the mode control to supported values', () => {
    expect(ENUM_OPTIONS['power.prevent_sleep.mode']).toEqual(['system', 'display'])
  })
})

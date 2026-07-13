import { describe, expect, it } from 'vitest'

import { SECTIONS } from './constants'

describe('settings sections', () => {
  it('exposes both model-facing security controls in Safety', () => {
    const safety = SECTIONS.find((section) => section.label === 'Safety')

    expect(safety?.keys).toEqual(
      expect.arrayContaining([
        'security.redact_secrets',
        'security.computer_use_safety_guidance'
      ])
    )
  })
})

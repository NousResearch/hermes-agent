import { describe, expect, it } from 'vitest'

import { withActive } from './model-settings'

// A Radix <Select> shows a blank trigger when its `value` matches no
// <SelectItem>. `withActive` guarantees the controlled value is always
// representable so a config-only / custom model never renders blank.
describe('withActive', () => {
  const curated = ['hermes-4', 'hermes-4-mini']

  it('prepends a custom model missing from the curated list', () => {
    expect(withActive(curated, 'anthropic/claude-opus-4.7')).toEqual(['anthropic/claude-opus-4.7', ...curated])
  })

  it('leaves the list untouched when the active model is already curated', () => {
    expect(withActive(curated, 'hermes-4')).toEqual(curated)
  })

  it('does not inject an empty active value', () => {
    expect(withActive(curated, '')).toEqual(curated)
  })
})

import { describe, expect, it } from 'vitest'

import { fromItemValue, toItemValue, withActive } from './model-select'

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

// The "Custom model…" action row shares the Radix value namespace with the
// model rows. Any whitespace-free slug is a legal model id here, so a model
// must round-trip and never decode as the action.
describe('item values', () => {
  it.each(['hermes-4', '__custom__', 'custom', 'model:', 'model:custom'])('round-trips %s as a model', slug => {
    expect(fromItemValue(toItemValue(slug))).toBe(slug)
  })

  it('keeps an empty selection empty so the trigger shows its placeholder', () => {
    expect(toItemValue('')).toBe('')
  })
})

import { REASONING_EFFORTS } from '@hermes/shared'
import { describe, expect, it } from 'vitest'

import { ko } from '@/i18n/ko'

// The rendered Korean panel is covered in `custom-endpoints-settings.test.tsx`, which
// already owns the module mocks for it. These are catalog-level invariants, so they stay
// free of mocks and of the component.
describe('Korean catalog invariants', () => {
  it.each(REASONING_EFFORTS)('keeps the %s reasoning effort as the token providers use', value => {
    expect(ko.shell.modelOptions[value]).toBe(value)
  })

  it('still translates Fast, which is an ordinary toggle rather than an effort token', () => {
    expect(ko.shell.modelOptions.fast).toBe('빠름')
  })

  it('translates the Browser settings section, which falls back to its English label without a catalog entry', () => {
    expect(ko.settings.sections.browser).toBe('브라우저')
  })
})

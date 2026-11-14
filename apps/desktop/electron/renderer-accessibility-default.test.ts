import assert from 'node:assert/strict'

import { test } from 'vitest'

import { shouldEnableRendererAccessibility } from './renderer-accessibility'

test('default startup leaves native assistive technology detection in control', () => {
  for (const platform of ['darwin', 'win32'] as const) {
    for (const value of [undefined, '', '   ']) {
      assert.equal(
        shouldEnableRendererAccessibility({ HERMES_DESKTOP_RENDERER_ACCESSIBILITY: value }, platform),
        false,
        'startup must not force accessibility without an explicit opt-in'
      )
    }

    assert.equal(shouldEnableRendererAccessibility({ HERMES_DESKTOP_RENDERER_ACCESSIBILITY: '1' }, platform), true)
    assert.equal(shouldEnableRendererAccessibility({ HERMES_DESKTOP_RENDERER_ACCESSIBILITY: '0' }, platform), false)
  }
})

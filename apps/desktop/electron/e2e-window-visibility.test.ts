import { expect, test } from 'vitest'

import { E2E_HIDDEN_WINDOW_ENV, shouldKeepE2EWindowsHidden } from './e2e-window-visibility'

test('keeps E2E windows hidden only when the internal switch is explicit', () => {
  expect(shouldKeepE2EWindowsHidden({})).toBe(false)
  expect(shouldKeepE2EWindowsHidden({ [E2E_HIDDEN_WINDOW_ENV]: '0' })).toBe(false)
  expect(shouldKeepE2EWindowsHidden({ [E2E_HIDDEN_WINDOW_ENV]: '1' })).toBe(true)
})

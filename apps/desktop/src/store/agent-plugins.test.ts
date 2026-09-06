import { expect, it } from 'vitest'

import { isDesktopRelevantPlugin } from './agent-plugins'

it('keeps opt-in bundled plugins discoverable before and after enabling', () => {
  const row = { name: 'optional-native', key: 'optional-native', version: '1', description: '', source: 'bundled', default_enabled: false }

  for (const status of ['not enabled', 'enabled', 'disabled'] as const) {
    expect(isDesktopRelevantPlugin({ ...row, status })).toBe(true)
    expect(isDesktopRelevantPlugin({ ...row, default_enabled: true, status })).toBe(false)
  }
})

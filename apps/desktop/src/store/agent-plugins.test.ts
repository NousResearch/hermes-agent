import { expect, it, vi } from 'vitest'

import { installAgentPlugin, isDesktopRelevantPlugin } from './agent-plugins'

it('keeps opt-in bundled plugins discoverable before and after enabling', () => {
  const row = {
    name: 'optional-native',
    key: 'optional-native',
    version: '1',
    description: '',
    source: 'bundled',
    default_enabled: false
  }

  for (const status of ['not enabled', 'enabled', 'disabled'] as const) {
    expect(isDesktopRelevantPlugin({ ...row, status })).toBe(true)
    expect(isDesktopRelevantPlugin({ ...row, default_enabled: true, status })).toBe(false)
  }
})

it('preserves installed-but-not-enabled setup refusal so the UI can offer enable, not reclone', async () => {
  const request = vi.fn().mockRejectedValue(
    Object.assign(new Error('Review setup'), {
      data: {
        ok: false,
        status: 'consent_required',
        installed: true,
        plugin_name: 'native-fixture',
        error: 'Review setup',
        consent: { key: 'native-fixture', hermes_home: '/fixture/profile', revision: 'v1' }
      }
    })
  )

  const result = await installAgentPlugin(request, { identifier: 'owner/native-fixture', enable: true })
  expect(result).toMatchObject({ ok: false, installed: true, pluginName: 'native-fixture', error: 'Review setup' })
  expect(request).toHaveBeenCalledOnce()
})

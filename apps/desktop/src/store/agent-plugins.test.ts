import { expect, it, vi } from 'vitest'

import { installAgentPlugin } from './agent-plugins'

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

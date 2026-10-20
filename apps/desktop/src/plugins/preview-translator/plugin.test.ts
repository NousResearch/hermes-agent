import { afterEach, describe, expect, it, vi } from 'vitest'

import { discoverBundledPlugins } from '@/contrib/plugins'
import { $pluginDecisions, $pluginRecords, setPluginEnabled } from '@/contrib/plugins-store'
import { registry } from '@/contrib/registry'

vi.mock('@/contrib/runtime-loader', () => ({ watchRuntimePlugins: vi.fn() }))

afterEach(async () => {
  await setPluginEnabled('preview-translator', false)
})

describe('bundled Preview Translator plugin', () => {
  it('inventories off by default and contributes the status-bar chip only when enabled', async () => {
    $pluginDecisions.set({ accent: false, kanban: false, 'hermes-bots': false, radio: false })
    discoverBundledPlugins()

    expect($pluginRecords.get()['preview-translator']).toMatchObject({ kind: 'bundled', status: 'disabled' })
    expect(registry.getArea('statusBar.right').some(item => item.source === 'plugin:preview-translator')).toBe(false)

    await setPluginEnabled('preview-translator', true)
    expect($pluginRecords.get()['preview-translator'].status).toBe('loaded')
    expect(registry.getArea('statusBar.right').some(item => item.source === 'plugin:preview-translator')).toBe(true)

    await setPluginEnabled('preview-translator', false)
    expect(registry.getArea('statusBar.right').some(item => item.source === 'plugin:preview-translator')).toBe(false)
  })
})

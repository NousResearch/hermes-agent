import { afterEach, describe, expect, it, vi } from 'vitest'

import { $pluginUpdateAllRunning, isPluginUpdateAllRunning, updateAllAgentPlugins } from './plugin-update-all'

describe('plugin update-all running state', () => {
  afterEach(() => $pluginUpdateAllRunning.set(new Set()))

  it('marks only the profile whose run is in flight', async () => {
    let finish!: (value: unknown) => void
    const request = vi.fn(() => new Promise(resolve => (finish = resolve))) as never
    const run = updateAllAgentPlugins(request, 'work')

    expect(isPluginUpdateAllRunning($pluginUpdateAllRunning.get(), 'work')).toBe(true)
    expect(isPluginUpdateAllRunning($pluginUpdateAllRunning.get(), null)).toBe(false)
    expect(isPluginUpdateAllRunning($pluginUpdateAllRunning.get(), 'home')).toBe(false)

    finish({ results: [] })
    await run
    expect(isPluginUpdateAllRunning($pluginUpdateAllRunning.get(), 'work')).toBe(false)
  })
})

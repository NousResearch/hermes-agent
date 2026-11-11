import { afterEach, describe, expect, it, vi } from 'vitest'

import type { GatewayRequest } from './agent-plugins'
import {
  $pluginAutoUpdate,
  $pluginUpdateAllRunning,
  isPluginUpdateAllRunning,
  loadPluginAutoUpdate,
  updateAllAgentPlugins
} from './plugin-update-all'

describe('plugin update-all running state', () => {
  afterEach(() => {
    $pluginUpdateAllRunning.set(new Set())
    $pluginAutoUpdate.set(null)
  })

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

  it('does not mark a same-named agent on another gateway as updating', async () => {
    const pending = Promise.withResolvers<unknown>()

    const request = Object.assign(
      vi.fn(() => pending.promise),
      { ownerKey: 'gateway-a/work' }
    ) as GatewayRequest

    const run = updateAllAgentPlugins(request, 'work')
    expect(isPluginUpdateAllRunning($pluginUpdateAllRunning.get(), 'work', 'gateway-a/work')).toBe(true)
    expect(isPluginUpdateAllRunning($pluginUpdateAllRunning.get(), 'work', 'gateway-b/work')).toBe(false)
    pending.resolve({ results: [] })
    await run
    expect(isPluginUpdateAllRunning($pluginUpdateAllRunning.get(), 'work', 'gateway-a/work')).toBe(false)
  })

  it('ignores an old gateway auto-update response after another owner loads', async () => {
    const pending = Promise.withResolvers<unknown>()

    const requestA = Object.assign(
      vi.fn(() => pending.promise),
      { ownerKey: 'gateway-a/work' }
    ) as GatewayRequest

    const requestB = Object.assign(
      vi.fn(async () => ({ auto_apply: true, check_hours: 12 })),
      { ownerKey: 'gateway-b/work' }
    ) as GatewayRequest

    const first = loadPluginAutoUpdate(requestA, 'work')
    await loadPluginAutoUpdate(requestB, 'work')
    pending.resolve({ auto_apply: false, check_hours: 24 })
    await first
    expect($pluginAutoUpdate.get()).toEqual({
      autoApply: true,
      checkHours: 12,
      profile: 'work',
      ownerKey: 'gateway-b/work'
    })
  })
})

import { beforeEach, expect, it, vi } from 'vitest'

import type { GatewayRequest } from './agent-plugins'
import {
  $pluginMarketplaces,
  $pluginMarketplaceScope,
  addPluginMarketplace,
  loadPluginMarketplaces
} from './plugin-marketplaces'

beforeEach(() => {
  $pluginMarketplaces.set([])
  $pluginMarketplaceScope.set(null)
})

it('ignores an old profile result even after switching A → B → A', async () => {
  let finishOld!: (value: unknown) => void

  const request = vi.fn((_, params?: Record<string, unknown>) => {
    if (params?.profile === 'A' && !finishOld) {
      return new Promise(resolve => {
        finishOld = resolve
      })
    }

    return Promise.resolve({ marketplaces: [{ id: params?.profile, name: `current ${params?.profile}`, entries: [] }] })
  })

  const old = loadPluginMarketplaces(request as GatewayRequest, 'A')
  await loadPluginMarketplaces(request as GatewayRequest, 'B')
  await loadPluginMarketplaces(request as GatewayRequest, 'A')
  finishOld({ marketplaces: [{ id: 'old', name: 'stale', entries: [] }] })
  await old

  expect($pluginMarketplaceScope.get()).toBe('A')
  expect($pluginMarketplaces.get().map(item => item.name)).toEqual(['current A'])
})

it('adding a marketplace never sends an install request', async () => {
  const request = vi.fn(async (_, params?: Record<string, unknown>) =>
    params?.action === 'marketplace_add' ? { ok: true } : { marketplaces: [] }
  )

  await loadPluginMarketplaces(request as GatewayRequest, 'team')
  expect(await addPluginMarketplace(request as GatewayRequest, 'https://github.com/team/plugins', 'team')).toBe(true)
  expect(request).toHaveBeenCalledWith('plugins.manage', {
    action: 'marketplace_add',
    url: 'https://github.com/team/plugins',
    profile: 'team'
  })
  expect(request.mock.calls.some(([, params]) => params?.action === 'install')).toBe(false)
})

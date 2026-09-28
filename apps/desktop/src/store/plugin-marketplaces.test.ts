import { beforeEach, expect, it, vi } from 'vitest'

import type { GatewayRequest } from './agent-plugins'
import {
  $pluginMarketplaces,
  $pluginMarketplaceScope,
  addPluginMarketplace,
  loadPluginMarketplaces,
  marketplaceScopeKey,
  removePluginMarketplace
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

  const old = loadPluginMarketplaces(request as GatewayRequest, 'A', 'one')
  await loadPluginMarketplaces(request as GatewayRequest, 'B', 'one')
  await loadPluginMarketplaces(request as GatewayRequest, 'A', 'one')
  finishOld({ marketplaces: [{ id: 'old', name: 'stale', entries: [] }] })
  await old

  expect($pluginMarketplaceScope.get()).toBe(marketplaceScopeKey('one', 'A'))
  expect($pluginMarketplaces.get().map(item => item.name)).toEqual(['current A'])
})

it('separates same-profile connections and ignores a stale removal after switching back', async () => {
  let finish!: (value: unknown) => void
  const first = vi.fn((_, params?: Record<string, unknown>) =>
    params?.action === 'marketplace_remove' ? new Promise(resolve => { finish = resolve }) :
      Promise.resolve({ marketplaces: [{ id: 'one', name: 'first', entries: [] }] })
  )
  const second = vi.fn(async () => ({ marketplaces: [{ id: 'two', name: 'second', entries: [] }] }))
  await loadPluginMarketplaces(first as GatewayRequest, 'team', 'connection-one')
  const pending = removePluginMarketplace(first as GatewayRequest, 'one', 'team', 'connection-one')
  await loadPluginMarketplaces(second as GatewayRequest, 'team', 'connection-two')
  expect($pluginMarketplaces.get()[0]?.name).toBe('second')
  await loadPluginMarketplaces(first as GatewayRequest, 'team', 'connection-one')
  finish({ ok: true, removed: true })
  await pending
  expect($pluginMarketplaces.get()[0]?.name).toBe('first')
  expect(first).toHaveBeenCalledWith('plugins.manage', { action: 'marketplace_remove', source_id: 'one', profile: 'team' })
  expect(first.mock.calls.filter(([, params]) => params?.action === 'marketplaces')).toHaveLength(2)
})

it('adding a marketplace never sends an install request', async () => {
  const request = vi.fn(async (_, params?: Record<string, unknown>) =>
    params?.action === 'marketplace_add' ? { ok: true } : { marketplaces: [] }
  )

  await loadPluginMarketplaces(request as GatewayRequest, 'team', 'one')
  expect(await addPluginMarketplace(request as GatewayRequest, 'https://github.com/team/plugins', 'team', 'one')).toBe(true)
  expect(request).toHaveBeenCalledWith('plugins.manage', {
    action: 'marketplace_add',
    url: 'https://github.com/team/plugins',
    profile: 'team'
  })
  expect(request.mock.calls.some(([, params]) => params?.action === 'install')).toBe(false)
})

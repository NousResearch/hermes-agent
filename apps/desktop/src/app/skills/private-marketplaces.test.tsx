import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $agentPlugins, type GatewayRequest } from '@/store/agent-plugins'
import {
  $pluginMarketplaceBusy,
  $pluginMarketplaces,
  $pluginMarketplaceScope,
  $pluginMarketplacesStatus,
  type PluginMarketplace
} from '@/store/plugin-marketplaces'

import { PrivateMarketplaces } from './private-marketplaces'

const marketplace: PluginMarketplace = {
  entries: [
    {
      compatible: true,
      description: 'A private plugin.',
      display_name: 'Private Plugin',
      incompatibility_reason: '',
      maintainer: 'Example',
      name: 'private-plugin',
      source_id: 'market-1',
      source_name: 'Private Market',
      version: '1.2.0'
    }
  ],
  id: 'market-1',
  name: 'Private Market',
  available: true,
  stale: false
}

const requestMock = vi.fn(async (_method: string, params?: Record<string, unknown>) => {
  if (params?.action === 'marketplaces') {
    return { marketplaces: [marketplace] }
  }

  if (params?.action === 'marketplace_add') {
    return { marketplace, ok: true }
  }

  if (params?.action === 'update') {
    return { ok: true, unchanged: false }
  }

  if (params?.action === 'install') {
    return { ok: true, plugin_name: 'private-plugin' }
  }

  return { marketplaces: [marketplace], plugins: [] }
})

const request = requestMock as unknown as GatewayRequest

describe('PrivateMarketplaces', () => {
  beforeEach(() => {
    requestMock.mockClear()
    $pluginMarketplaces.set([])
    $pluginMarketplaceScope.set(null)
    $pluginMarketplacesStatus.set('idle')
    $pluginMarketplaceBusy.set(null)
    $agentPlugins.set([])
  })

  afterEach(cleanup)

  it('installs through the selected backend without exposing clone coordinates', async () => {
    render(
      <PrivateMarketplaces
        installed={[]}
        officialCatalog={<div>Official Catalog</div>}
        profile="workbot"
        request={request}
        scopeKey="workbot"
      />
    )

    ;(await screen.findByRole('tab', { name: 'Private Market' })).click()
    expect(await screen.findByText('Private Plugin')).toBeTruthy()
    screen.getByRole('button', { name: 'Install' }).click()
    expect(await screen.findByText('Install Private Plugin?')).toBeTruthy()
    screen.getAllByRole('button', { name: 'Install' }).at(-1)?.click()

    await waitFor(() => {
      expect(requestMock).toHaveBeenCalledWith(
        'plugins.manage',
        expect.objectContaining({
          action: 'install',
          identifier: '',
          marketplace_id: 'market-1',
          marketplace_plugin_name: 'private-plugin',
          profile: 'workbot'
        })
      )
    })
  })

  it('shows Update when the installed plugin subtree changed', async () => {
    const installed = [
      {
        description: '',
        key: 'private-plugin',
        marketplace_id: 'market-1',
        marketplace_name: 'Private Market',
        marketplace_plugin_name: 'private-plugin',
        name: 'private-plugin',
        source: 'git',
        status: 'enabled' as const,
        update_available: true,
        version: '1.1.0'
      }
    ]

    render(
      <PrivateMarketplaces
        installed={installed}
        officialCatalog={<div>Official Catalog</div>}
        profile={null}
        request={request}
        scopeKey="default"
      />
    )

    ;(await screen.findByRole('tab', { name: 'Private Market' })).click()
    const update = await screen.findByRole('button', { name: 'Update' })
    update.click()

    await waitFor(() =>
      expect(requestMock).toHaveBeenCalledWith(
        'plugins.manage',
        expect.objectContaining({ action: 'update', key: 'private-plugin' })
      )
    )
  })

  it('closes an install confirmation when the profile changes', async () => {
    const view = render(
      <PrivateMarketplaces
        installed={[]}
        officialCatalog={<div>Official Catalog</div>}
        profile="first"
        request={request}
        scopeKey="one::first"
      />
    )

    ;(await screen.findByRole('tab', { name: 'Private Market' })).click()
    await screen.findByText('Private Plugin')
    screen.getByRole('button', { name: 'Install' }).click()
    expect(await screen.findByText('Install Private Plugin?')).toBeTruthy()

    view.rerender(
      <PrivateMarketplaces
        installed={[]}
        officialCatalog={<div>Official Catalog</div>}
        profile="second"
        request={request}
        scopeKey="two::second"
      />
    )
    await waitFor(() => expect(screen.queryByText('Install Private Plugin?')).toBeNull())
  })

  it('does not publish an install continuation after the backend changes', async () => {
    let resolveInstall: ((value: unknown) => void) | undefined

    const delayedRequestMock = vi.fn(async (_method: string, params?: Record<string, unknown>) => {
      if (params?.action === 'install') {
        return await new Promise(resolve => (resolveInstall = resolve))
      }

      return { marketplaces: [marketplace], plugins: [] }
    })

    const delayedRequest = delayedRequestMock as unknown as GatewayRequest

    const view = render(
      <PrivateMarketplaces
        installed={[]}
        officialCatalog={<div>Official Catalog</div>}
        profile="same"
        request={delayedRequest}
        scopeKey="one::same"
      />
    )

    ;(await screen.findByRole('tab', { name: 'Private Market' })).click()
    await screen.findByText('Private Plugin')
    screen.getByRole('button', { name: 'Install' }).click()
    await screen.findByText('Install Private Plugin?')
    screen.getAllByRole('button', { name: 'Install' }).at(-1)?.click()
    await waitFor(() => expect(resolveInstall).toBeTypeOf('function'))

    view.rerender(
      <PrivateMarketplaces
        installed={[]}
        officialCatalog={<div>Official Catalog</div>}
        profile="same"
        request={delayedRequest}
        scopeKey="two::same"
      />
    )
    resolveInstall?.({ ok: true, plugin_name: 'private-plugin' })
    await waitFor(() => expect(screen.queryByText('Install Private Plugin?')).toBeNull())

    expect(delayedRequestMock.mock.calls.some(([, params]) => params?.action === 'list')).toBe(false)
  })

  it('adds a marketplace URL through the profile-scoped backend', async () => {
    render(
      <PrivateMarketplaces
        installed={[]}
        officialCatalog={<div>Official Catalog</div>}
        profile="workbot"
        request={request}
        scopeKey="workbot"
      />
    )

    screen.getByRole('button', { name: 'Add marketplace' }).click()
    const input = await screen.findByRole('textbox', { name: 'Marketplace repository URL' })
    fireEvent.change(input, {
      target: { value: 'https://github.com/example/marketplace' }
    })
    screen.getByRole('button', { name: 'Add marketplace' }).click()

    await waitFor(() =>
      expect(requestMock).toHaveBeenCalledWith(
        'plugins.manage',
        expect.objectContaining({
          action: 'marketplace_add',
          profile: 'workbot',
          url: 'https://github.com/example/marketplace'
        })
      )
    )
  })

  it('tabs between the official catalog and marketplace display names', async () => {
    const second: PluginMarketplace = {
      ...marketplace,
      id: 'market-2',
      name: 'Automation Lab',
      entries: [{ ...marketplace.entries[0], display_name: 'Automation Plugin', name: 'automation-plugin' }]
    }

    const tabsRequest = vi.fn(async () => ({
      marketplaces: [marketplace, second],
      plugins: []
    })) as unknown as GatewayRequest

    render(
      <PrivateMarketplaces
        installed={[]}
        officialCatalog={<div>Official Catalog</div>}
        profile={null}
        request={tabsRequest}
        scopeKey="default"
      />
    )

    expect(screen.getByText('Official Catalog')).toBeTruthy()
    ;(await screen.findByRole('tab', { name: 'Private Market' })).click()
    expect(await screen.findByText('Private Plugin')).toBeTruthy()
    expect(screen.queryByText('Automation Plugin')).toBeNull()

    screen.getByRole('tab', { name: 'Automation Lab' }).click()
    expect(await screen.findByText('Automation Plugin')).toBeTruthy()
    expect(screen.queryByText('Private Plugin')).toBeNull()
  })
})

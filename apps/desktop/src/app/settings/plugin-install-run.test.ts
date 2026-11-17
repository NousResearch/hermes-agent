import { afterEach, expect, it, vi } from 'vitest'

import { $agentPluginsByOwner, loadAgentPlugins, pluginInventoryKey } from '@/store/agent-plugins'

import { runPluginInstall } from './plugin-install-run'

afterEach(() => {
  $agentPluginsByOwner.set({})
})

it('refreshes the installed owner from a new list instead of coalescing with a pre-install read', async () => {
  let finishOld!: (value: { plugins: unknown[] }) => void
  let lists = 0

  const request = vi.fn(async (_method: string, params: Record<string, unknown>) => {
    if (params.action === 'install') {
      return { ok: true, plugin_name: 'weather' }
    }

    lists += 1

    return lists === 1
      ? new Promise(resolve => {
          finishOld = resolve
        })
      : { plugins: [{ key: 'weather', name: 'weather', catalog_name: 'weather', status: 'enabled' }] }
  })

  const oldList = loadAgentPlugins(request as never, 'workbot', 'local')

  const install = runPluginInstall(
    {
      request: {
        repo: 'https://example.org/weather',
        catalogName: 'weather',
        profile: { connectionId: 'local', profile: 'workbot' }
      },
      profile: 'workbot',
      probe: { agent: true, desktop: false },
      enable: true,
      force: false,
      remote: false
    } as never,
    {
      copy: { agentSuccess: (name: string) => `${name} installed` },
      navigate: vi.fn(),
      requestGateway: request as never
    } as never
  )

  await vi.waitFor(() => expect(lists).toBe(2))
  finishOld({ plugins: [] })
  expect(await install).toMatchObject({ status: 'done' })
  await oldList
  expect($agentPluginsByOwner.get()[pluginInventoryKey('local', 'workbot')].map(row => row.catalog_name)).toEqual([
    'weather'
  ])
})

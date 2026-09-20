import { afterEach, beforeEach, expect, it, vi } from 'vitest'

vi.mock('@/store/gateway', () => ({ requestGatewayForAgent: vi.fn() }))
vi.mock('@/store/notifications', () => ({ notifyError: vi.fn() }))

import { setApiRequestConnection, setApiRequestProfile } from '@/api/client'
import { queryClient } from '@/lib/query-client'
import { requestGatewayForAgent } from '@/store/gateway'

import {
  $agentPlugins,
  discoverInstalledMemoryProvider,
  installAgentPlugin,
  loadAgentPlugins,
  toggleAgentPlugin
} from './agent-plugins'

const owner = (connectionId: string) => ({ connectionId, profile: 'alpha' })
const api = vi.fn()
window.hermesDesktop = { api } as never

function activate(connectionId: string | null, profile: string) {
  setApiRequestConnection(connectionId)
  setApiRequestProfile(profile)
}

beforeEach(() => {
  vi.clearAllMocks()
  queryClient.clear()
  activate('source-a', 'alpha')
})
afterEach(() => activate(null, ''))

it('routes mutations to their owner and refreshes that owner, not the foreground', async () => {
  vi.mocked(requestGatewayForAgent).mockImplementation(async (connection, _profile, _method, params) =>
    params?.action === 'list' ? { plugins: [{ name: connection }] } : { ok: true }
  )
  activate('source-b', 'alpha')
  await loadAgentPlugins(owner('source-b'))
  expect((await installAgentPlugin({ identifier: 'fixture/plugin', profile: owner('source-a') })).ok).toBe(true)
  await toggleAgentPlugin('fixture', true, 'failed', owner('source-a'))

  for (const action of ['install', 'toggle', 'list']) {
    const params = expect.objectContaining({ action, profile: 'alpha' })
    expect(requestGatewayForAgent).toHaveBeenCalledWith('source-a', 'alpha', 'plugins.manage', params)
  }

  expect($agentPlugins.get()).toEqual([{ name: 'source-b' }])
})

it('discovers installed code for the captured owner after the foreground moved', async () => {
  api.mockResolvedValue({ active: 'fixture', providers: [{ name: 'fixture', status: 'needs_config' }] })
  const discovering = discoverInstalledMemoryProvider(owner('source-a'), 'fixture')
  activate('source-b', 'alpha')
  expect(await discovering).toBe(true)
  expect(api).toHaveBeenCalledExactlyOnceWith(
    expect.objectContaining({ path: '/api/memory', profile: 'alpha', connectionId: 'source-a' })
  )
})

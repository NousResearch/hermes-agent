import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $apiRequestScope } from '@/api/client'
import { pluginSettingsRouteHref } from '@/contrib/settings-pages'
import { queryClient } from '@/lib/query-client'
import { $agentPlugins, $agentPluginsStatus } from '@/store/agent-plugins'
import { $activeGatewayProfile, $profiles } from '@/store/profile'
import { $settingsScopeOverride } from '@/store/settings-scope'
import { $agentSidebarChoice } from '@/store/simplified-agent-mode'

import { SettingsView } from './entry'

const plugin = {
  description: '', key: 'image_gen/fal', name: 'Fal', source: 'user', status: 'enabled', version: '1',
  settings_schema: [{ description: '', key: 'region', label: 'Region', required: false, type: 'string', value: 'us' }]
}
const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
  if (method === 'plugins.manage') {
    return params?.action === 'list' ? { plugins: [plugin] } : { ok: true, plugin }
  }

  return {}
})
vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway })
}))

interface Request {
  path: string
  connectionId?: string
  profile?: string
  method?: string
  body?: unknown
}
const api = vi.fn(async ({ path, method }: Request): Promise<unknown> => {
  const pathname = path.split('?')[0]

  if (pathname === '/api/skills' || pathname === '/api/tools/toolsets') {return []}
  if (pathname === '/api/cron/jobs') {return method === 'POST' ? { id: 'saved', enabled: true } : []}
  if (pathname === '/api/messaging/platforms') {return { platforms: [] }}
  if (pathname === '/api/profiles/sessions') {return { sessions: [], total: 0, limit: 5, offset: 0 }}
  if (pathname.endsWith('/soul')) {return { content: 'Owner instructions' }}
  if (pathname === '/api/profiles') {
    return {
      current: 'default',
      profiles: [{
        name: 'default', is_default: true, path: '/test/hermes', has_env: false,
        model: null, provider: null, skill_count: 0
      }]
    }
  }

  return {}
})
const originalDesktop = window.hermesDesktop

beforeEach(() => {
  queryClient.clear()
  api.mockClear()
  requestGateway.mockClear()
  window.hermesDesktop = { api } as unknown as typeof window.hermesDesktop
  $profiles.set([])
  $activeGatewayProfile.set('default')
  $apiRequestScope.set({ connectionId: 'beta', profile: 'default' })
  $agentPlugins.set([])
  $agentPluginsStatus.set('idle')
  $agentSidebarChoice.set(true)
})
afterEach(() => {
  cleanup()
  queryClient.clear()
  window.hermesDesktop = originalDesktop
  $apiRequestScope.set({ profile: null, connectionId: null })
  $settingsScopeOverride.set(null)
  $agentSidebarChoice.set(false)
})

function mount(path: string) {
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[path]}>
        <SettingsView onClose={() => {}} />
      </MemoryRouter>
    </QueryClientProvider>
  )
}

it.each(['', '&connection=beta'])('pins every real overview request and routine write to one native owner (%s)', async connection => {
  mount(`/settings?agentProfile=default${connection}&tab=plugins&agent=image_gen%2Ffal`)
  await screen.findByText('Owner instructions')
  const paths = ['/api/config', '/api/skills', '/api/tools/toolsets', '/api/cron/jobs?profile=default']

  await waitFor(() => {
    for (const path of paths) {
      const request = api.mock.calls.find(([request]) => request.path === path)?.[0]

      expect(request).toBeDefined()
      // An absent connection would dispatch through the native primary instead.
      expect(request?.connectionId).toBe('beta')
      expect(request?.profile).toBe('default')
    }
  })
  fireEvent.click(screen.getByRole('button', { name: 'Add routine' }))
  fireEvent.change(screen.getByLabelText('Schedule'), { target: { value: '0 9 * * *' } })
  fireEvent.change(screen.getByLabelText('Prompt'), { target: { value: 'Review notes' } })
  fireEvent.click(screen.getByRole('button', { name: 'Save' }))
  await waitFor(() => expect(api.mock.calls.some(([request]) => request.method === 'POST')).toBe(true))
  const write = api.mock.calls.find(([request]) => request.method === 'POST')![0]

  expect(write.connectionId).toBe('beta')
  expect(write.profile).toBe('default')
  act(() => $agentSidebarChoice.set(false))
  expect(await screen.findByLabelText(/Region/)).toBeTruthy()
  expect(screen.queryByText('Owner instructions')).toBeNull()
  act(() => $agentSidebarChoice.set(true))
  expect(await screen.findByText('Owner instructions')).toBeTruthy()
  act(() => $apiRequestScope.setKey('connectionId', 'alpha'))
  expect(await screen.findByRole('button', { name: 'Open agent' })).toBeTruthy()
  expect(screen.queryByText('Owner instructions')).toBeNull()
  act(() => $agentSidebarChoice.set(false))
  expect(screen.getByRole('button', { name: 'Open agent' })).toBeTruthy()
  act(() => $agentSidebarChoice.set(true))
  expect(screen.getByRole('button', { name: 'Open agent' })).toBeTruthy()
})

it.each([false, true])('opens and saves the existing plugin route with redesign=%s', async redesigned => {
  $agentSidebarChoice.set(redesigned)
  mount(pluginSettingsRouteHref({ agent: plugin.key }, 'default'))
  const region = await screen.findByLabelText(/Region/)

  expect(screen.queryByRole('button', { name: 'Open agent' })).toBeNull()
  fireEvent.change(region, { target: { value: 'eu' } })
  fireEvent.click(screen.getByRole('button', { name: 'Save settings' }))
  await waitFor(() => expect(requestGateway).toHaveBeenCalledWith('plugins.manage', {
    action: 'settings', key: 'image_gen/fal', profile: 'default', values: { region: 'eu' }
  }))
})

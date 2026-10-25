import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { $activeConnectionId } from '@/store/connections'
import { setInterfaceMode } from '@/store/interface-mode'
import { $activeGatewayProfile, $profiles } from '@/store/profile'
import { $connection, $selectedStoredSessionId } from '@/store/session'
import { $settingsScopeOverride } from '@/store/settings-scope'
import type { ProfileInfo } from '@/types/hermes'

import { SettingsView } from './index'

const profiles: ProfileInfo[] = ['default', 'research'].map(name => ({
  has_env: false,
  is_default: name === 'default',
  model: '',
  name,
  path: '',
  provider: '',
  skill_count: 0
}))

afterEach(() => {
  cleanup()
  $settingsScopeOverride.set(null)
  $connection.set(null)
  $activeGatewayProfile.set('default')
  $selectedStoredSessionId.set(null)
  $profiles.set([])
  setInterfaceMode('advanced')
  vi.unstubAllGlobals()
})

it('edits capabilities on the selected settings owner without switching the workspace, and returns on Escape', async () => {
  const api = vi.fn(async ({ path }: { path: string }) => {
    if (path === '/api/profiles') {
      return { profiles, active: 'default' }
    }

    if (path === '/api/skills/hub/official') {
      return { skills: [] }
    }

    if (path === '/api/skills' || path === '/api/tools/toolsets') {
      return []
    }

    return {}
  })

  vi.stubGlobal('hermesDesktop', { api })
  setInterfaceMode('simple')
  $connection.set({
    connectionId: 'device-one',
    mode: 'remote',
    baseUrl: '',
    wsUrl: '',
    token: '',
    logs: [],
    isFullscreen: false,
    nativeOverlayWidth: 0,
    windowButtonPosition: null
  })
  $activeGatewayProfile.set('default')
  $selectedStoredSessionId.set('task-one')
  $profiles.set(profiles)
  const onClose = vi.fn()
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <MemoryRouter initialEntries={['/settings?tab=capabilities']}>
      <QueryClientProvider client={client}>
        <SettingsView onClose={onClose} />
      </QueryClientProvider>
    </MemoryRouter>
  )
  await waitFor(() =>
    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({
        connectionId: 'device-one',
        profile: 'default',
        path: '/api/skills'
      })
    )
  )
  fireEvent.click(screen.getByRole('button', { name: 'research' }))
  await waitFor(() =>
    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({
        connectionId: 'device-one',
        profile: 'research',
        path: '/api/skills'
      })
    )
  )
  expect($activeGatewayProfile.get()).toBe('default')
  expect($activeConnectionId.get()).toBe('device-one')
  expect($selectedStoredSessionId.get()).toBe('task-one')
  expect(screen.getByRole('button', { name: 'Back to workspace' })).toBeTruthy()
  act(() => fireEvent.keyDown(window, { key: 'Escape' }))
  expect(onClose).toHaveBeenCalledOnce()
  client.clear()
})

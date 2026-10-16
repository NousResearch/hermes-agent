import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type * as PluginStore from '@/contrib/plugins-store'
import { $pluginRecords, setPluginEnabled } from '@/contrib/plugins-store'
import type * as HermesApi from '@/hermes'
import { getToolsets, profileScopeKey, setApiRequestConnection, setToolsetEnabled } from '@/hermes'
import { queryClient } from '@/lib/query-client'
import { $agentPlugins, $agentPluginsStatus } from '@/store/agent-plugins'
import { notify, notifyError } from '@/store/notifications'
import type { ToolsetInfo } from '@/types/hermes'

import { PluginsTab } from './plugins-tab'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  getToolsets: vi.fn(),
  setToolsetEnabled: vi.fn()
}))
vi.mock('@/contrib/plugins-store', async importOriginal => ({
  ...(await importOriginal<typeof PluginStore>()),
  setPluginEnabled: vi.fn(async () => undefined)
}))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))
const requestGateway = vi.fn(async (_method: string, _params?: unknown) => ({ plugins: [] }))
vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway })
}))

const scopeA = { connectionId: 'remote-a', profile: 'planner' }
const scopeB = { connectionId: 'remote-b', profile: 'planner' }

const kanban = (enabled = false): ToolsetInfo => ({
  configured: true,
  description: 'Board tools',
  enabled,
  label: 'Kanban',
  name: 'kanban',
  tools: ['kanban_create']
})

const agentSwitch = () => screen.getByRole('switch', { name: 'Agent in planner: Kanban' }) as HTMLButtonElement

beforeEach(() => {
  vi.clearAllMocks()
  setApiRequestConnection(scopeA.connectionId)
  queryClient.clear()
  $agentPlugins.set([])
  $agentPluginsStatus.set('ready')
  $pluginRecords.set({ kanban: { id: 'kanban', name: 'Kanban', kind: 'bundled', status: 'loaded' } })
  vi.mocked(getToolsets).mockResolvedValue([kanban()])
  vi.mocked(setToolsetEnabled).mockResolvedValue({ ok: true, name: 'kanban', enabled: true })
})
afterEach(() => {
  cleanup()
  queryClient.clear()
  setApiRequestConnection(null)
})

it.each(['explicit', 'legacy'] as const)(
  'keeps Desktop separate and binds an in-flight tool grant to its %s scope',
  async mode => {
    const profileA = mode === 'legacy' ? scopeA.profile : scopeA
    let finish!: (value: { ok: boolean; name: string; enabled: boolean }) => void
    const save = new Promise<{ ok: boolean; name: string; enabled: boolean }>(resolve => {
      finish = resolve
    })
    vi.mocked(setToolsetEnabled).mockReturnValue(save)
    const { rerender } = render(<PluginsTab profile={profileA} scopeLabel="planner" />)
    await waitFor(() => expect(agentSwitch().disabled).toBe(false))
    expect(getToolsets).toHaveBeenCalledWith(profileA)
    expect(setToolsetEnabled).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('switch', { name: 'Desktop: Kanban' }))
    expect(setPluginEnabled).toHaveBeenCalledWith('kanban', false)
    expect(setToolsetEnabled).not.toHaveBeenCalled()
    fireEvent.click(agentSwitch())
    setApiRequestConnection(scopeB.connectionId)
    await waitFor(() => expect(setToolsetEnabled).toHaveBeenCalledWith('kanban', true, scopeA))
    expect(agentSwitch().getAttribute('aria-checked')).toBe('true')
    rerender(<PluginsTab profile={scopeB} scopeLabel="planner" />)
    await waitFor(() => {
      expect(getToolsets).toHaveBeenCalledWith(scopeB)
      expect(agentSwitch().getAttribute('aria-checked')).toBe('false')
      expect(agentSwitch().disabled).toBe(false)
    })
    await act(async () => {
      finish({ ok: true, name: 'kanban', enabled: true })
      await save
    })
    await waitFor(() =>
      expect(notify).toHaveBeenCalledWith(
        expect.objectContaining({
          kind: 'success',
          message: expect.stringContaining('Open a new chat')
        })
      )
    )
    expect(agentSwitch().getAttribute('aria-checked')).toBe('false')
    expect(queryClient.getQueryData<ToolsetInfo[]>(['toolsets-list', profileScopeKey(scopeB)])?.[0].enabled).toBe(false)
    expect(setToolsetEnabled).toHaveBeenCalledTimes(1)
    expect(requestGateway.mock.calls.every(call => call[0] === 'plugins.manage')).toBe(true)
  }
)

it.each(['missing', 'read-error', 'write-error', 'invalid-receipt'] as const)(
  'does not pretend the tools were enabled on %s and leaves the board usable',
  async mode => {
    if (mode === 'missing') {
      vi.mocked(getToolsets).mockResolvedValue([])
    }

    if (mode === 'read-error') {
      vi.mocked(getToolsets).mockRejectedValueOnce(new Error('offline'))
    }

    if (mode === 'write-error') {
      vi.mocked(setToolsetEnabled).mockRejectedValue(new Error('write denied'))
    }

    if (mode === 'invalid-receipt') {
      vi.mocked(setToolsetEnabled).mockResolvedValue({ ok: false, name: 'kanban', enabled: true })
    }
    render(<PluginsTab profile={scopeA} scopeLabel="planner" />)
    expect(screen.getByRole('switch', { name: 'Desktop: Kanban' }).getAttribute('aria-checked')).toBe('true')

    if (mode === 'missing') {
      await screen.findByText('Update backend')
      expect(agentSwitch().disabled).toBe(true)
      expect(setToolsetEnabled).not.toHaveBeenCalled()
    } else if (mode === 'read-error') {
      const retry = await screen.findByRole('button', { name: 'Retry' })
      expect(agentSwitch().disabled).toBe(true)
      expect(screen.queryByText('Update backend')).toBeNull()
      fireEvent.click(retry)
      await waitFor(() => expect(agentSwitch().disabled).toBe(false))
    } else {
      await waitFor(() => expect(agentSwitch().disabled).toBe(false))
      fireEvent.click(agentSwitch())
      await waitFor(() => expect(notifyError).toHaveBeenCalled())
      await waitFor(() => expect(agentSwitch().getAttribute('aria-checked')).toBe('false'))
      expect(notify).not.toHaveBeenCalled()
    }

    expect(setPluginEnabled).not.toHaveBeenCalled()
  }
)

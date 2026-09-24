import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

const { requestGateway, rawGateway } = vi.hoisted(() => ({
  rawGateway: { current: undefined as undefined | { request: ReturnType<typeof vi.fn> } },
  requestGateway: vi.fn()
}))

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway, gateway: rawGateway.current })
}))

import { $pluginRecords } from '@/contrib/plugins-store'
import {
  $agentPluginBusy,
  $agentPlugins,
  $agentPluginsError,
  $agentPluginsStatus,
  type AgentPluginRow
} from '@/store/agent-plugins'
import { $activeGatewayProfile } from '@/store/profile'

import { PluginsTab } from './plugins-tab'

const row: AgentPluginRow = {
  name: 'Native fixture',
  key: 'native-fixture',
  version: '1',
  description: 'An opt-in bundled native plugin',
  source: 'bundled',
  default_enabled: false,
  status: 'disabled'
}

const proposal = (revision = 'v1') => ({
  status: 'consent_required',
  setup: {
    revision,
    ready: false,
    summary: 'Install fixture runtime',
    details: ['https://example.com/runtime.zip', '/fixture/profile/bin/runtime']
  },
  consent: { key: row.key, hermes_home: '/fixture/profile', revision }
})

const reviewError = (revision = 'v1', message = 'Review setup') =>
  Object.assign(new Error(message), { data: proposal(revision) })

beforeEach(() => {
  requestGateway.mockReset()
  // The replacement page loads its inventory on mount; keep that read
  // separate from the reviewed mutation (unlike the removed settings screen).
  requestGateway.mockResolvedValue({ plugins: [row] })
  rawGateway.current = undefined
  $pluginRecords.set({})
  $agentPlugins.set([row])
  $agentPluginsStatus.set('ready')
  $agentPluginsError.set(null)
  $agentPluginBusy.set(null)
  $activeGatewayProfile.set('default')
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

const renderPlugins = async (profile: string | null = null) => {
  const view = render(<PluginsTab profile={profile} />)
  await act(async () => undefined)

  return view
}

it('reviews exact native setup, keeps enable off while busy, surfaces failure and retries', async () => {
  await renderPlugins('work')
  requestGateway.mockRejectedValueOnce(reviewError())
  fireEvent.click(screen.getByRole('switch', { name: 'Agent: Native fixture' }))
  expect(await screen.findByText('Install fixture runtime')).toBeTruthy()
  expect(screen.getByText('https://example.com/runtime.zip')).toBeTruthy()
  expect(screen.getByText('/fixture/profile/bin/runtime')).toBeTruthy()
  let fail!: (reason: Error) => void
  requestGateway.mockImplementationOnce(
    () =>
      new Promise((_resolve, reject) => {
        fail = reject
      })
  )
  fireEvent.click(screen.getByRole('button', { name: 'Set up and enable' }))
  await waitFor(() =>
    expect(requestGateway).toHaveBeenLastCalledWith(
      'plugins.manage',
      { action: 'toggle', key: row.key, enable: true, profile: 'work', setup_consent: proposal().consent },
      360000
    )
  )
  expect($agentPlugins.get()[0].status).toBe('disabled')
  expect((screen.getByRole('button', { name: 'Cancel' }) as HTMLButtonElement).disabled).toBe(true)
  fail(new Error('Runtime verification failed; install fixture-library and retry'))
  expect(await screen.findByText('Runtime verification failed; install fixture-library and retry')).toBeTruthy()
  requestGateway.mockResolvedValueOnce({ ok: true, plugin: { ...row, status: 'enabled' } })
  fireEvent.click(screen.getByRole('button', { name: 'Set up and enable' }))
  await waitFor(() => expect($agentPlugins.get()[0].status).toBe('enabled'))
  await waitFor(() =>
    expect(screen.getByRole('switch', { name: 'Agent: Native fixture' }).getAttribute('aria-checked')).toBe('true')
  )
})

it('canceling setup does not send consent', async () => {
  await renderPlugins()
  requestGateway.mockRejectedValueOnce(reviewError())
  fireEvent.click(screen.getByRole('switch', { name: 'Agent: Native fixture' }))
  fireEvent.click(await screen.findByRole('button', { name: 'Cancel' }))
  expect(requestGateway.mock.calls.filter(([, params]) => params.action === 'toggle')).toHaveLength(1)
  expect($agentPlugins.get()[0].status).toBe('disabled')
})

it.each(['active', 'selected'] as const)('drops pending consent when the %s profile changes', async kind => {
  const view = await renderPlugins()
  requestGateway.mockRejectedValueOnce(reviewError())
  fireEvent.click(screen.getByRole('switch', { name: 'Agent: Native fixture' }))
  expect(await screen.findByText('Install fixture runtime')).toBeTruthy()

  if (kind === 'active') {
    act(() => $activeGatewayProfile.set('other'))
  } else {
    view.rerender(<PluginsTab profile="other" />)
  }

  await waitFor(() => expect(screen.queryByText('Install fixture runtime')).toBeNull())
  expect(requestGateway.mock.calls.filter(([, params]) => params.action === 'toggle')).toHaveLength(1)
})

it('pins consent to the captured raw gateway and requires another click for a changed revision', async () => {
  const original = vi.fn().mockRejectedValueOnce(reviewError())
  const other = vi.fn()
  rawGateway.current = { request: original }
  await renderPlugins('work')
  fireEvent.click(screen.getByRole('switch', { name: 'Agent: Native fixture' }))
  expect(await screen.findByText('v1')).toBeTruthy()
  rawGateway.current = { request: other }
  act(() => $agentPlugins.set([{ ...row, description: 'New connection row' }]))
  original.mockRejectedValueOnce(reviewError('v2', 'Revision changed; review again'))
  fireEvent.click(screen.getByRole('button', { name: 'Set up and enable' }))
  expect(await screen.findByText('Revision changed; review again')).toBeTruthy()
  expect(screen.getByText('v2')).toBeTruthy()
  expect(original).toHaveBeenCalledTimes(2)
  expect(other).not.toHaveBeenCalled()
  expect(requestGateway.mock.calls.every(([, params]) => params.action === 'list')).toBe(true)
  original.mockResolvedValueOnce({ ok: true, plugin: { ...row, status: 'enabled' } })
  fireEvent.click(screen.getByRole('button', { name: 'Set up and enable' }))
  await waitFor(() =>
    expect(original).toHaveBeenLastCalledWith(
      'plugins.manage',
      { action: 'toggle', key: row.key, enable: true, profile: 'work', setup_consent: proposal('v2').consent },
      360000
    )
  )
})

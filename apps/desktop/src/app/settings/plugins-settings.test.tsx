import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const { requestGateway, getProfiles, rawGateway } = vi.hoisted(() => ({
  rawGateway: { current: undefined as undefined | { request: ReturnType<typeof vi.fn> } },
  requestGateway: vi.fn(),
  getProfiles: vi.fn<() => Promise<{ profiles: { name: string; is_default: boolean }[] }>>(async () => ({
    profiles: []
  }))
}))

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway, gateway: rawGateway.current })
}))

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getProfiles
}))

import { $pluginRecords } from '@/contrib/plugins-store'
import { queryClient } from '@/lib/query-client'
import {
  $agentPluginBusy,
  $agentPlugins,
  $agentPluginsError,
  $agentPluginsStatus,
  type AgentPluginRow
} from '@/store/agent-plugins'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection, $gatewayState } from '@/store/session'

import { PluginsSettings } from './plugins-settings'

const legacyRow = {
  name: 'Legacy plugin',
  version: '0.20.0',
  description: 'Returned by a pre-key backend',
  source: 'user',
  status: 'disabled'
} satisfies AgentPluginRow

const renderSettings = () =>
  render(
    <QueryClientProvider client={queryClient}>
      <PluginsSettings />
    </QueryClientProvider>
  )

beforeEach(() => {
  requestGateway.mockReset()
  rawGateway.current = undefined
  getProfiles.mockReset()
  getProfiles.mockResolvedValue({ profiles: [] })
  queryClient.clear()
  $pluginRecords.set({})
  $agentPlugins.set([legacyRow])
  $agentPluginsStatus.set('ready')
  $agentPluginsError.set(null)
  $agentPluginBusy.set(null)
  $gatewayState.set('idle')
  $connection.set(null)
  $activeGatewayProfile.set('default')
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('PluginsSettings', () => {
  it('renders and searches plugin rows returned without a canonical key', () => {
    renderSettings()

    expect(screen.getByText('Legacy plugin')).toBeTruthy()

    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'pre-key' } })

    expect(screen.getByText('Legacy plugin')).toBeTruthy()
  })

  it('renders keyless rows read-only instead of falling back to name-addressed toggles', () => {
    // Name-addressed toggles flip every same-named plugin across category
    // dirs (image_gen/fal vs video_gen/fal) — the reason toggles moved to
    // canonical keys. A pre-contract-v6 row must never reach the RPC.
    renderSettings()

    const toggle = screen.getByRole('switch', { name: 'Enable Legacy plugin' })

    expect(toggle.hasAttribute('disabled') || toggle.getAttribute('aria-disabled') === 'true').toBe(true)

    fireEvent.click(toggle)

    expect(requestGateway).not.toHaveBeenCalledWith('plugins.manage', expect.objectContaining({ action: 'toggle' }))
  })

  it('keeps duplicate-named keyless rows distinct (no React key collision)', () => {
    const sibling = {
      ...legacyRow,
      description: 'A second plugin category with the same legacy name'
    }

    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => undefined)

    $agentPlugins.set([legacyRow, sibling])

    renderSettings()

    expect(screen.getAllByRole('switch', { name: 'Enable Legacy plugin' })).toHaveLength(2)
    expect(screen.getByText(sibling.description)).toBeTruthy()
    expect(consoleError.mock.calls.flat().join(' ')).not.toContain('same key')
  })

  it('keeps using the canonical key when the backend provides one', async () => {
    const keyedRow = { ...legacyRow, key: 'image_gen/legacy' }

    $agentPlugins.set([keyedRow])
    requestGateway.mockResolvedValue({ ok: true, plugin: { ...keyedRow, status: 'enabled' } })

    renderSettings()
    fireEvent.click(screen.getByRole('switch', { name: 'Enable Legacy plugin' }))

    await waitFor(() =>
      expect(requestGateway).toHaveBeenCalledWith('plugins.manage', {
        action: 'toggle',
        key: 'image_gen/legacy',
        enable: true
      })
    )
  })

  it('hides repo-bundled built-ins and keeps the count pill in sync', () => {
    // The Agent plugins section is the control panel for plugins the USER
    // installed — built-ins (browser backends, cron providers, model
    // providers…) ship enabled-by-default and are configured elsewhere.
    $agentPlugins.set([
      legacyRow,
      { ...legacyRow, name: 'browserbase', key: 'browser/browserbase', source: 'bundled' },
      { ...legacyRow, name: 'chronos', key: 'cron_providers/chronos', source: 'bundled' },
      { ...legacyRow, name: 'deepinfra', key: 'model-providers/deepinfra', source: 'bundled' }
    ])

    renderSettings()

    expect(screen.getByText('Legacy plugin')).toBeTruthy()
    expect(screen.queryByText('browserbase')).toBeNull()
    expect(screen.queryByText('chronos')).toBeNull()
    expect(screen.queryByText('deepinfra')).toBeNull()
    // Count pill reflects the filtered list, not the raw RPC row count.
    expect(screen.getByText('1 installed', { exact: false })).toBeTruthy()
  })

  it('hides legacy other-surface categories even when the backend omits source', () => {
    // Older backends may not report source reliably — the key-prefix
    // fallback still hides categories other surfaces own.
    $agentPlugins.set([{ ...legacyRow, name: 'deepinfra', key: 'model-providers/deepinfra', source: 'user' }])

    renderSettings()

    expect(screen.queryByText('deepinfra')).toBeNull()
  })

  it('shows no profile selector with a single profile', async () => {
    getProfiles.mockResolvedValue({ profiles: [{ name: 'default', is_default: true }] })

    renderSettings()

    await waitFor(() => expect(getProfiles).toHaveBeenCalled())
    expect(screen.queryByText('Applies to:')).toBeNull()
  })

  it('lists the active profile scope without a profile param and reloads scoped on change', async () => {
    getProfiles.mockResolvedValue({
      profiles: [
        { name: 'default', is_default: true },
        { name: 'work', is_default: false }
      ]
    })
    requestGateway.mockResolvedValue({ plugins: [legacyRow] })
    $gatewayState.set('open')

    renderSettings()

    // Active profile scope: no profile param — older backends unchanged.
    await waitFor(() => expect(requestGateway).toHaveBeenCalledWith('plugins.manage', { action: 'list' }))
    await waitFor(() => expect(screen.getByText('Applies to:')).toBeTruthy())
  })

  it('sends toggles through the selected profile scope', async () => {
    // jsdom's scrollIntoView is missing/non-functional; Radix Select calls it
    // when the dropdown opens.
    Element.prototype.scrollIntoView = vi.fn()

    const keyedRow = { ...legacyRow, key: 'image_gen/legacy' }

    getProfiles.mockResolvedValue({
      profiles: [
        { name: 'default', is_default: true },
        { name: 'work', is_default: false }
      ]
    })
    requestGateway.mockImplementation(async (method: string, params?: Record<string, unknown>) => {
      if (params?.action === 'list') {
        return { plugins: [keyedRow] }
      }

      return { ok: true, plugin: { ...keyedRow, status: 'enabled' } }
    })
    $gatewayState.set('open')

    renderSettings()

    await waitFor(() => expect(screen.getByText('Applies to:')).toBeTruthy())

    // Select the non-active profile scope.
    fireEvent.click(screen.getByRole('combobox'))
    fireEvent.click(await screen.findByText('work'))

    await waitFor(() =>
      expect(requestGateway).toHaveBeenCalledWith('plugins.manage', { action: 'list', profile: 'work' })
    )

    fireEvent.click(screen.getByRole('switch', { name: 'Enable Legacy plugin' }))

    await waitFor(() =>
      expect(requestGateway).toHaveBeenCalledWith('plugins.manage', {
        action: 'toggle',
        key: 'image_gen/legacy',
        enable: true,
        profile: 'work'
      })
    )
  })
})

it('reviews exact native setup, keeps enable off while busy, surfaces failure and retries', async () => {
  const row = { ...legacyRow, key: 'native-fixture' }

  const proposal = {
    ok: false,
    status: 'consent_required',
    error: 'Review setup',
    setup: {
      revision: 'v1-hash',
      ready: false,
      summary: 'Install fixture runtime',
      details: ['https://example.com/runtime.zip', '/fixture/profile/bin/runtime']
    },
    consent: { key: row.key, hermes_home: '/fixture/profile', revision: 'v1-hash' }
  }

  $agentPlugins.set([row])
  requestGateway.mockRejectedValueOnce(Object.assign(new Error('Review setup'), { data: proposal }))
  renderSettings()
  fireEvent.click(screen.getByRole('switch', { name: 'Enable Legacy plugin' }))
  expect(await screen.findByText('Install fixture runtime')).toBeTruthy()
  expect(screen.getByText('https://example.com/runtime.zip')).toBeTruthy()
  expect(screen.getByText('/fixture/profile/bin/runtime')).toBeTruthy()
  expect(requestGateway).toHaveBeenCalledTimes(1)
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
      {
        action: 'toggle',
        key: row.key,
        enable: true,
        setup_consent: proposal.consent
      },
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
})

it('canceling setup does not send consent', async () => {
  $agentPlugins.set([{ ...legacyRow, key: 'native-fixture' }])
  requestGateway.mockRejectedValueOnce(
    Object.assign(new Error('Review setup'), {
      data: {
        status: 'consent_required',
        setup: { revision: 'v1', ready: false, summary: 'Native setup', details: [] },
        consent: { key: 'native-fixture', hermes_home: '/fixture/profile', revision: 'v1' }
      }
    })
  )
  renderSettings()
  fireEvent.click(screen.getByRole('switch', { name: 'Enable Legacy plugin' }))
  fireEvent.click(await screen.findByRole('button', { name: 'Cancel' }))
  expect(requestGateway).toHaveBeenCalledTimes(1)
  expect($agentPlugins.get()[0].status).toBe('disabled')
})

it('drops a pending consent dialog when the active profile changes', async () => {
  const row = { ...legacyRow, key: 'native-fixture' }
  $agentPlugins.set([row])
  requestGateway.mockRejectedValueOnce(
    Object.assign(new Error('Review setup'), {
      data: {
        status: 'consent_required',
        setup: { revision: 'v1', ready: false, summary: 'Native setup', details: [] },
        consent: { key: row.key, hermes_home: '/fixture/profile', revision: 'v1' }
      }
    })
  )
  renderSettings()
  fireEvent.click(screen.getByRole('switch', { name: 'Enable Legacy plugin' }))
  expect(await screen.findByText('Native setup')).toBeTruthy()
  act(() => $activeGatewayProfile.set('other'))
  await waitFor(() => expect(screen.queryByText('Native setup')).toBeNull())
  expect(requestGateway).toHaveBeenCalledTimes(1)
})

it('keeps reviewed consent on the captured raw gateway and requires a second review for a changed revision', async () => {
  const row = { ...legacyRow, key: 'native-fixture' }

  const proposal = (revision: string) => ({
    status: 'consent_required',
    setup: { revision, ready: false, summary: 'Native setup', details: [] },
    consent: { key: row.key, hermes_home: '/fixture/profile', revision }
  })

  const original = vi.fn().mockRejectedValueOnce(Object.assign(new Error('Review setup'), { data: proposal('v1') }))
  const other = vi.fn()
  rawGateway.current = { request: original }
  $agentPlugins.set([row])
  renderSettings()
  fireEvent.click(screen.getByRole('switch', { name: 'Enable Legacy plugin' }))
  expect(await screen.findByText('v1')).toBeTruthy()
  rawGateway.current = { request: other }
  act(() => $agentPlugins.set([{ ...row, description: 'New connection row' }]))
  original.mockRejectedValueOnce(Object.assign(new Error('Revision changed; review again'), { data: proposal('v2') }))
  fireEvent.click(screen.getByRole('button', { name: 'Set up and enable' }))
  expect(await screen.findByText('Revision changed; review again')).toBeTruthy()
  expect(screen.getByText('v2')).toBeTruthy()
  expect(original).toHaveBeenCalledTimes(2)
  expect(other).not.toHaveBeenCalled()
  expect(requestGateway).not.toHaveBeenCalled()
  original.mockResolvedValueOnce({ ok: true, plugin: { ...row, status: 'enabled' } })
  fireEvent.click(screen.getByRole('button', { name: 'Set up and enable' }))
  await waitFor(() =>
    expect(original).toHaveBeenLastCalledWith(
      'plugins.manage',
      {
        action: 'toggle',
        key: row.key,
        enable: true,
        setup_consent: proposal('v2').consent
      },
      360000
    )
  )
})

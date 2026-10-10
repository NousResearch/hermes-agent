import { host } from '@hermes/plugin-sdk'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen } from '@testing-library/react'
import type { ReactElement } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { discoverBundledPlugins } from './plugins'
import { $pluginDecisions, $pluginRecords, setPluginEnabled } from './plugins-store'
import { registry } from './registry'

vi.mock('./runtime-loader', () => ({ watchRuntimePlugins: vi.fn() }))

const pluginId = 'mission-control'
// Other bundled plugins are outside this integration test.
const otherBundled = { accent: false, kanban: false, 'hermes-bots': false, radio: false }

function paneContribution() {
  return registry.getArea('panes').find(item => item.source === `plugin:${pluginId}`)
}

function paneElement(): ReactElement | null {
  return (paneContribution()?.render?.() ?? null) as ReactElement | null
}

function gatewayResponses() {
  const now = Date.now() / 1000

  const working = {
    id: 'sess-2',
    session_key: 'sess-2',
    current: false,
    title: '',
    preview: 'A long-running build is still streaming its transcript',
    model: 'nous/hermes-4-405b',
    status: 'working',
    message_count: 12,
    started_at: now - 600,
    last_active: now - 20
  }

  return {
    'session.list': {
      sessions: [
        {
          id: 'sess-1',
          session_key: 'sess-1',
          current: false,
          title: 'Weekly digest',
          preview: 'drafted the digest',
          model: 'nous/hermes-4-405b',
          status: 'idle',
          message_count: 6,
          started_at: now - 3600,
          last_active: now - 300
        },
        working
      ]
    },
    'session.active_list': { sessions: [working] },
    'cron.manage': {
      jobs: [
        {
          job_id: 'job-1',
          name: 'Quota watchdog',
          schedule: '0 * * * *',
          next_run_at: new Date(Date.now() + 3_600_000).toISOString(),
          enabled: true,
          state: 'scheduled',
          last_status: 'ok'
        },
        {
          job_id: 'job-2',
          name: 'Nightly backup',
          schedule: '0 3 * * *',
          next_run_at: new Date(Date.now() + 7_200_000).toISOString(),
          enabled: false,
          state: 'paused',
          last_status: 'error'
        }
      ],
      gateway_running: true
    }
  }
}

afterEach(async () => {
  await setPluginEnabled(pluginId, false)
  vi.restoreAllMocks()
})

describe('bundled Mission control plugin', () => {
  it('inventories off by default and follows the ordinary live enable/disable lifecycle', async () => {
    $pluginDecisions.set(otherBundled)
    const stylesBefore = document.head.querySelectorAll('style').length

    discoverBundledPlugins()
    expect($pluginRecords.get()[pluginId]).toMatchObject({ kind: 'bundled', status: 'disabled' })
    expect(paneElement()).toBeNull()
    expect(document.head.querySelectorAll('style').length).toBe(stylesBefore)

    await setPluginEnabled(pluginId, true)
    expect($pluginRecords.get()[pluginId].status).toBe('loaded')
    const contribution = paneContribution()
    expect(contribution).toMatchObject({ id: 'mission-control:pane', area: 'panes' })
    expect((contribution?.data as { dock?: unknown } | undefined)?.dock).toEqual({ pane: 'workspace', pos: 'right' })
    expect(document.head.querySelectorAll('style').length).toBe(stylesBefore + 1)

    await setPluginEnabled(pluginId, false)
    expect(paneElement()).toBeNull()
    expect(document.head.querySelectorAll('style').length).toBe(stylesBefore)
  })

  it('renders the roster with the working split, cron footer and click-to-open', async () => {
    $pluginDecisions.set(otherBundled)
    const responses = gatewayResponses()

    vi.spyOn(host, 'request').mockImplementation(
      (async (method: string) => responses[method as keyof typeof responses] ?? {}) as typeof host.request
    )
    const open = vi.spyOn(host, 'openSession').mockImplementation((() => Promise.resolve()) as typeof host.openSession)

    discoverBundledPlugins()
    await setPluginEnabled(pluginId, true)

    const queries = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(<QueryClientProvider client={queries}>{paneElement()}</QueryClientProvider>)

    expect(await screen.findByText('Weekly digest')).toBeTruthy()
    expect(screen.getByText('A long-running build is still streaming its transcript')).toBeTruthy()
    expect(screen.getByText('working')).toBeTruthy()
    expect(screen.getByText('cron — 1 scheduled · 1 paused')).toBeTruthy()
    expect(screen.getByText('Quota watchdog')).toBeTruthy()
    expect(screen.getByText('running')).toBeTruthy()

    fireEvent.click(screen.getByText('Weekly digest'))
    expect(open).toHaveBeenCalledWith('sess-1', { intent: 'stack' })
  })
})

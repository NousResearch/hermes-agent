import { act, cleanup, render, screen, within } from '@testing-library/react'
import { useEffect, useState } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { setApiRequestConnection } from '@/api/client'
import { PLUGIN_SETTINGS_AREA } from '@/contrib/plugin-settings'
import { $pluginRecords } from '@/contrib/plugins-store'
import { registry } from '@/contrib/registry'
import type { PluginSettingsContributionProps } from '@/sdk'
import { $agentPlugins, $agentPluginsStatus } from '@/store/agent-plugins'

import { PluginsTab } from './plugins-tab'

const requestGateway = vi.fn(async () => ({ plugins: [] }))
vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway })
}))
const dispose: (() => void)[] = []
beforeEach(() => {
  $pluginRecords.set({ demo: { id: 'demo', name: 'Demo', kind: 'disk', status: 'loaded' } })
  $agentPlugins.set([])
  $agentPluginsStatus.set('ready')
})
afterEach(() => {
  cleanup()
  dispose.splice(0).forEach(fn => fn())
  setApiRequestConnection(null)
})

it('uses the actual REST connection tag for a named default selection and follows source changes', async () => {
  dispose.push(
    registry.register({
      id: 'tagged',
      source: 'plugin:demo',
      area: PLUGIN_SETTINGS_AREA,
      data: {
        render: ({ scope }: PluginSettingsContributionProps) => (
          <span>
            Settings {scope.connectionId}/{scope.profile}
          </span>
        )
      }
    })
  )
  setApiRequestConnection('source-a')
  const view = render(<PluginsTab profile="work" />)
  await act(async () => {})
  expect(screen.getByText('Settings source-a/work')).toBeTruthy()
  await act(async () => {
    setApiRequestConnection('source-b')
  })
  expect(await screen.findByText('Settings source-b/work')).toBeTruthy()
  expect(screen.queryByText('Settings source-a/work')).toBeNull()
  await act(async () => {
    view.rerender(<PluginsTab profile={null} />)
  })
  expect(screen.queryByText(/^Settings /)).toBeNull()
  act(() => {
    setApiRequestConnection(null)
  })
  await act(async () => {
    view.rerender(<PluginsTab profile="work" />)
  })
  expect(screen.queryByText(/^Settings /)).toBeNull()
})

it('mounts optional settings inside the owning package row with its selected scope', async () => {
  dispose.push(
    registry.register({
      id: 'demo:settings',
      source: 'plugin:demo',
      area: PLUGIN_SETTINGS_AREA,
      data: {
        render: ({ scope }: PluginSettingsContributionProps) => (
          <button type="button">
            Settings for {scope.connectionId}/{scope.profile}
          </button>
        )
      }
    })
  )
  await act(async () => {
    render(<PluginsTab profile={{ connectionId: 'remote-a', profile: 'work' }} />)
  })
  const row = screen.getByRole('switch', { name: 'Desktop: Demo' }).closest('[role="row"]')!
  expect(within(row as HTMLElement).getByRole('button', { name: 'Settings for remote-a/work' })).toBeTruthy()
})

it('isolates settings to exact loaded plugin ownership and removes them on unload', async () => {
  $pluginRecords.set({
    demo: { id: 'demo', name: 'Demo', kind: 'disk', status: 'loaded' },
    disabled: { id: 'disabled', name: 'Disabled', kind: 'disk', status: 'disabled' },
    broken: { id: 'broken', name: 'Broken', kind: 'disk', status: 'error' }
  })

  for (const source of ['plugin:demo', 'plugin:demo-extra', 'plugin:disabled', 'plugin:broken', 'core']) {
    dispose.push(
      registry.register({
        id: source,
        source,
        area: PLUGIN_SETTINGS_AREA,
        data: { render: () => <span>Settings {source}</span> }
      })
    )
  }

  dispose.push(
    registry.register({
      id: 'hidden',
      source: 'plugin:demo',
      area: PLUGIN_SETTINGS_AREA,
      enabled: false,
      data: { render: () => <span>Settings hidden</span> }
    })
  )
  await act(async () => {
    render(<PluginsTab profile={{ connectionId: 'local', profile: 'work' }} />)
  })
  expect(screen.getAllByText(/^Settings /).map(el => el.textContent)).toEqual(['Settings plugin:demo'])
  act(() => dispose[0]())
  expect(screen.queryByText(/^Settings /)).toBeNull()
  expect(screen.getByRole('switch', { name: 'Desktop: Demo' })).toBeTruthy()
})

it('remounts scoped dialogs and retires old async work when the selected connection changes', async () => {
  const captured: PluginSettingsContributionProps['scope'][] = []
  const finish: (() => void)[] = []

  function Settings({ scope }: PluginSettingsContributionProps) {
    const [ready, setReady] = useState(false)
    useEffect(() => {
      let live = true
      captured.push(scope)
      finish.push(() => {
        if (live) {
          setReady(true)
        }
      })

      return () => {
        live = false
      }
      // The probe must stay mount-scoped: a dependency refresh would hide stale dialogs.
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [])

    return (
      <div role="dialog">
        {scope.connectionId}/{scope.profile}: {ready ? 'ready' : 'pending'}
      </div>
    )
  }

  dispose.push(
    registry.register({ id: 'dialog', source: 'plugin:demo', area: PLUGIN_SETTINGS_AREA, data: { render: Settings } })
  )
  const selected = { connectionId: 'remote-a', profile: 'work' }
  const view = render(<PluginsTab profile={selected} />)
  await act(async () => {})
  await act(async () => {
    view.rerender(<PluginsTab profile={{ connectionId: 'remote-b', profile: 'work' }} />)
  })
  act(() => finish[0]())
  expect(screen.getByRole('dialog').textContent).toBe('remote-b/work: pending')
  expect(captured).toHaveLength(2)
  expect(Object.isFrozen(captured[0])).toBe(true)
  selected.connectionId = 'changed-by-caller'
  expect(captured[0].connectionId).toBe('remote-a')
  act(() => finish[1]())
  expect(screen.getByRole('dialog').textContent).toBe('remote-b/work: ready')
})

it('contains a throwing contribution without losing sibling settings or the enable switch', async () => {
  const error = vi.spyOn(console, 'error').mockImplementation(() => {})

  try {
    dispose.push(
      registry.registerMany([
        {
          id: 'throwing',
          source: 'plugin:demo',
          area: PLUGIN_SETTINGS_AREA,
          data: {
            render: () => {
              throw new Error('settings failure')
            }
          }
        },
        {
          id: 'healthy',
          source: 'plugin:demo',
          area: PLUGIN_SETTINGS_AREA,
          data: { render: () => <span>Healthy settings</span> }
        }
      ])
    )
    await act(async () => {
      render(<PluginsTab profile={{ connectionId: 'local', profile: 'work' }} />)
    })
    expect(screen.getByText('Healthy settings')).toBeTruthy()
    expect(screen.getByRole('switch', { name: 'Desktop: Demo' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'throwing' })).toBeTruthy()
  } finally {
    error.mockRestore()
  }
})

it('withholds settings for implicit or incomplete scopes without inventing a local route', async () => {
  dispose.push(
    registry.register({
      id: 'scope',
      source: 'plugin:demo',
      area: PLUGIN_SETTINGS_AREA,
      data: {
        render: ({ scope }: PluginSettingsContributionProps) => (
          <span>
            Settings {scope.connectionId}/{scope.profile}
          </span>
        )
      }
    })
  )
  const view = render(<PluginsTab profile={{ connectionId: 'unknown-remote', profile: 'work' }} />)
  await act(async () => {})
  expect(screen.getByText('Settings unknown-remote/work')).toBeTruthy()

  for (const profile of [
    undefined,
    null,
    'work',
    {},
    { connectionId: 'local' },
    { profile: 'work' },
    { connectionId: ' ', profile: 'work' },
    { connectionId: 'local', profile: ' ' }
  ]) {
    await act(async () => {
      view.rerender(<PluginsTab profile={profile} />)
    })
    expect(screen.queryByText(/^Settings /)).toBeNull()
    expect(screen.getByRole('switch', { name: 'Desktop: Demo' })).toBeTruthy()
  }
})

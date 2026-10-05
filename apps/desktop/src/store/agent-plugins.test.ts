import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  $agentPluginBusy,
  $agentPlugins,
  $agentPluginsStatus,
  type AgentPluginRow,
  installAgentPlugin,
  isDesktopRelevantPlugin,
  loadAgentPlugins,
  normalizeAgentPluginRow,
  saveAgentPluginSettings,
  toggleAgentPlugin
} from './agent-plugins'

const row = (partial: Partial<AgentPluginRow>): AgentPluginRow =>
  ({ name: partial.key ?? 'x', status: 'enabled', ...partial }) as AgentPluginRow

afterEach(() => vi.useRealTimers())

describe('installAgentPlugin', () => {
  it('waits for a slow successful install instead of reporting the generic 30s timeout', async () => {
    vi.useFakeTimers()

    const request = vi.fn(
      <T>(_method: string, _params?: Record<string, unknown>, timeoutMs = 30_000): Promise<T> =>
        new Promise((resolve, reject) => {
          const deadline = setTimeout(
            () => reject(new Error(`request timed out after ${timeoutMs / 1000}s: plugins.manage`)),
            timeoutMs
          )

          setTimeout(() => {
            clearTimeout(deadline)
            resolve({ ok: true, plugin_name: 'demo' } as T)
          }, 45_000)
        })
    )

    const install = installAgentPlugin(request as never, { identifier: 'demo', profile: 'research' })

    await vi.advanceTimersByTimeAsync(45_000)

    expect(await install).toMatchObject({ ok: true, pluginName: 'demo' })
    expect(request).toHaveBeenCalledWith(
      'plugins.manage',
      expect.objectContaining({ action: 'install', profile: 'research' }),
      expect.any(Number)
    )
  })

  it('marks a client timeout as an unknown install outcome', async () => {
    const request = vi.fn(async () => {
      throw new Error('request timed out after 120s: plugins.manage')
    })

    expect(await installAgentPlugin(request as never, { identifier: 'demo' })).toMatchObject({
      ok: false,
      timedOut: true
    })
  })
})

describe('normalizeAgentPluginRow', () => {
  it('treats an absent servers field as an empty full snapshot', () => {
    const previous = normalizeAgentPluginRow(
      row({
        key: 'example-plugin',
        servers: [{ name: 'example-server', sentence: '', state: 'connected' }],
        source: 'user'
      })
    )

    const next = normalizeAgentPluginRow(row({ key: 'example-plugin', source: 'user' }))

    expect(previous.servers).toHaveLength(1)
    expect(next.servers).toEqual([])
  })
})

describe('isDesktopRelevantPlugin (#98861)', () => {
  it('hides ordinary built-ins but always lists user installs', () => {
    expect(isDesktopRelevantPlugin(row({ key: 'platforms/discord', source: 'bundled' }))).toBe(false)

    // User installs are unaffected either way.
    expect(isDesktopRelevantPlugin(row({ key: 'my-plugin', source: 'user' }))).toBe(true)
  })
})

describe('saveAgentPluginSettings (#46600, #87934)', () => {
  it('writes values through plugins.manage settings and secrets ONLY through the credential writer', async () => {
    $agentPlugins.set([row({ key: 'demo', source: 'user' })])
    const refreshed = row({ key: 'demo', settings_schema: [], source: 'user' })
    const request = vi.fn(async () => ({ ok: true, plugin: refreshed }))
    const writeSecret = vi.fn(async () => ({ ok: true }))

    const ok = await saveAgentPluginSettings(request as never, {
      failMessage: 'fail',
      key: 'demo',
      profile: 'workbot',
      secrets: { DEMO_API_KEY: 'sk-1', DEMO_OTHER: '' },
      values: { retries: 2 },
      writeSecret
    })

    expect(ok).toBe(true)
    expect(request).toHaveBeenCalledWith('plugins.manage', {
      action: 'settings',
      key: 'demo',
      profile: 'workbot',
      values: { retries: 2 }
    })
    // Blank secret = keep; the secret value never appears in any RPC payload.
    expect(writeSecret).toHaveBeenCalledTimes(1)
    expect(writeSecret).toHaveBeenCalledWith('DEMO_API_KEY', 'sk-1')
    expect(JSON.stringify(request.mock.calls)).not.toContain('sk-1')
    expect($agentPlugins.get()[0].settings_schema).toEqual([])
  })
})


describe('loadAgentPlugins scope isolation', () => {
  it('does not reuse or render local plugin state after switching to the same profile on another gateway', async () => {
    let resolveLocalRefresh!: (value: { plugins: AgentPluginRow[] }) => void
    let resolveHomelab!: (value: { plugins: AgentPluginRow[] }) => void

    const localRow = row({ key: 'local-only', name: 'Local only', source: 'user' })
    const homelabRow = row({ key: 'homelab-only', name: 'Homelab only', source: 'user' })

    const localRequest = vi
      .fn()
      .mockResolvedValueOnce({ plugins: [localRow] })
      .mockImplementationOnce(
        () =>
          new Promise(resolve => {
            resolveLocalRefresh = resolve
          })
      )
    const homelabRequest = vi.fn(
      () =>
        new Promise(resolve => {
          resolveHomelab = resolve
        })
    )

    await loadAgentPlugins(localRequest as never, 'default', 'local::default')

    expect($agentPlugins.get()).toEqual([expect.objectContaining({ key: 'local-only' })])
    expect($agentPluginsStatus.get()).toBe('ready')

    const pendingLocal = loadAgentPlugins(localRequest as never, 'default', 'local::default')
    const pendingHomelab = loadAgentPlugins(homelabRequest as never, 'default', 'homelab::default')

    expect(localRequest).toHaveBeenCalledTimes(2)
    expect(homelabRequest).toHaveBeenCalledTimes(1)
    expect(homelabRequest).toHaveBeenCalledWith(
      'plugins.manage',
      expect.objectContaining({ action: 'list', profile: 'default' })
    )
    expect($agentPlugins.get()).toEqual([])
    expect($agentPluginsStatus.get()).toBe('loading')

    resolveLocalRefresh({ plugins: [localRow] })
    await pendingLocal

    expect($agentPlugins.get()).toEqual([])
    expect($agentPluginsStatus.get()).toBe('loading')

    resolveHomelab({ plugins: [homelabRow] })
    await pendingHomelab

    expect($agentPlugins.get()).toEqual([expect.objectContaining({ key: 'homelab-only' })])
    expect($agentPluginsStatus.get()).toBe('ready')
  })
})


describe('agent plugin mutation scope isolation', () => {
  it('ignores a late mutation from the previous gateway and does not clear the new scope busy state', async () => {
    let resolveLocalToggle!: (value: { ok: boolean; plugin: AgentPluginRow }) => void
    let resolveHomelabToggle!: (value: { ok: boolean; plugin: AgentPluginRow }) => void

    const localRow = row({ key: 'shared-plugin', name: 'Shared plugin', source: 'user', status: 'disabled' })
    const homelabRow = row({ key: 'shared-plugin', name: 'Shared plugin', source: 'user', status: 'disabled' })

    const localRequest = vi.fn((_method: string, params?: Record<string, unknown>) => {
      if (params?.action === 'list') {
        return Promise.resolve({ plugins: [localRow] })
      }

      return new Promise(resolve => {
        resolveLocalToggle = resolve
      })
    })
    const homelabRequest = vi.fn((_method: string, params?: Record<string, unknown>) => {
      if (params?.action === 'list') {
        return Promise.resolve({ plugins: [homelabRow] })
      }

      return new Promise(resolve => {
        resolveHomelabToggle = resolve
      })
    })

    await loadAgentPlugins(localRequest as never, 'default', 'local::default')
    const pendingLocalToggle = toggleAgentPlugin(
      localRequest as never,
      'shared-plugin',
      true,
      'toggle failed',
      'default'
    )

    await loadAgentPlugins(homelabRequest as never, 'default', 'homelab::default')
    const pendingHomelabToggle = toggleAgentPlugin(
      homelabRequest as never,
      'shared-plugin',
      true,
      'toggle failed',
      'default'
    )

    resolveLocalToggle({
      ok: true,
      plugin: { ...localRow, status: 'enabled' }
    })
    await pendingLocalToggle

    expect($agentPlugins.get()).toEqual([expect.objectContaining({ key: 'shared-plugin', status: 'disabled' })])
    expect($agentPluginBusy.get()).toBe('shared-plugin')

    resolveHomelabToggle({
      ok: true,
      plugin: { ...homelabRow, status: 'enabled' }
    })
    await pendingHomelabToggle

    expect($agentPlugins.get()).toEqual([expect.objectContaining({ key: 'shared-plugin', status: 'enabled' })])
    expect($agentPluginBusy.get()).toBeNull()
  })

  it('keeps a mutation bound to its starting profile when the gateway request function is reused', async () => {
    let resolveDefaultToggle!: (value: { ok: boolean; plugin: AgentPluginRow }) => void

    const defaultRow = row({ key: 'shared-plugin', name: 'Shared plugin', source: 'user', status: 'disabled' })
    const researcherRow = row({ key: 'shared-plugin', name: 'Shared plugin', source: 'user', status: 'disabled' })

    const request = vi.fn((_method: string, params?: Record<string, unknown>) => {
      if (params?.action === 'list') {
        return Promise.resolve({
          plugins: [params.profile === 'researcher' ? researcherRow : defaultRow]
        })
      }

      return new Promise(resolve => {
        resolveDefaultToggle = resolve
      })
    })

    await loadAgentPlugins(request as never, 'default', 'local::default')
    const pendingDefaultToggle = toggleAgentPlugin(
      request as never,
      'shared-plugin',
      true,
      'toggle failed',
      'default'
    )

    await loadAgentPlugins(request as never, 'researcher', 'local::researcher')

    resolveDefaultToggle({
      ok: true,
      plugin: { ...defaultRow, status: 'enabled' }
    })
    await pendingDefaultToggle

    expect($agentPlugins.get()).toEqual([expect.objectContaining({ key: 'shared-plugin', status: 'disabled' })])
  })

})

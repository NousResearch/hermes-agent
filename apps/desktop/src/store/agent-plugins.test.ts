import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  $agentPlugins,
  $agentPluginsProfile,
  type AgentPluginRow,
  installAgentPlugin,
  isDesktopRelevantPlugin,
  normalizeAgentPluginRow,
  saveAgentPluginSettings,
  updateAgentPlugin
} from './agent-plugins'

const row = (partial: Partial<AgentPluginRow>): AgentPluginRow =>
  ({ name: partial.key ?? 'x', status: 'enabled', ...partial }) as AgentPluginRow

afterEach(() => {
  vi.useRealTimers()
  $agentPluginsProfile.set(undefined)
})

// A `plugins.manage` call that honours its deadline (30s unless the caller overrides it) like the
// real channel, but only answers after 45s; `list` refreshes answer at once.
const slowManageRequest = (result: unknown) =>
  vi.fn(
    <T>(_method: string, params?: Record<string, unknown>, timeoutMs = 30_000): Promise<T> =>
      new Promise((resolve, reject) => {
        if (params?.action === 'list') {
          resolve({ plugins: [] } as T)

          return
        }

        const deadline = setTimeout(
          () => reject(new Error(`request timed out after ${timeoutMs / 1000}s: plugins.manage`)),
          timeoutMs
        )

        setTimeout(() => {
          clearTimeout(deadline)
          resolve(result as T)
        }, 45_000)
      })
  )

describe('installAgentPlugin', () => {
  it('waits for a slow successful install instead of reporting the generic 30s timeout', async () => {
    vi.useFakeTimers()

    const request = slowManageRequest({ ok: true, plugin_name: 'demo' })

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

describe('updateAgentPlugin (#135565)', () => {
  it('waits for a slow catalog re-pin instead of reporting the generic 30s timeout', async () => {
    vi.useFakeTimers()

    const request = slowManageRequest({ ok: true })

    const update = updateAgentPlugin(request as never, 'demo', 'Update failed', 'research')

    await vi.advanceTimersByTimeAsync(45_000)

    expect(await update).toEqual({ kind: 'applied' })
    // The re-pin repeats an install's expensive work, so it carries the install budget, not the 30s default.
    expect(request).toHaveBeenCalledWith(
      'plugins.manage',
      expect.objectContaining({ action: 'update', name: 'demo', profile: 'research' }),
      120_000
    )
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
    $agentPluginsProfile.set('workbot')
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

  // The list is shared across profiles: a save that lands after the list moved
  // to another profile must neither patch that list nor refetch over it.
  it('leaves a list loaded for another profile alone', async () => {
    const other = row({ key: 'demo', source: 'user' })

    $agentPlugins.set([other])
    $agentPluginsProfile.set('other')
    const request = vi.fn(async () => ({ ok: true, plugin: row({ key: 'demo', settings_schema: [], source: 'user' }) }))

    const ok = await saveAgentPluginSettings(request as never, {
      failMessage: 'fail',
      key: 'demo',
      profile: 'workbot',
      secrets: {},
      values: { retries: 2 },
      writeSecret: vi.fn(async () => ({ ok: true }))
    })

    expect(ok).toBe(true)
    expect(request).toHaveBeenCalledTimes(1)
    expect($agentPlugins.get()).toEqual([other])
    expect($agentPluginsProfile.get()).toBe('other')
  })
})

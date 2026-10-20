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

describe('installAgentPlugin', () => {
  it.each(['install', 'update'])('waits for a slow successful %s instead of reporting a 30s failure', async action => {
    vi.useFakeTimers()

    const request = vi.fn(
      <T>(_method: string, _params?: Record<string, unknown>, timeoutMs = 30_000): Promise<T> =>
        new Promise((resolve, reject) => {
          if (_params?.action === 'list') {
            resolve({ ok: true, plugins: [] } as T)

            return
          }

          const deadline = setTimeout(
            () => reject(new Error(`request timed out after ${timeoutMs / 1000}s: plugins.manage`)),
            timeoutMs
          )

          setTimeout(() => {
            clearTimeout(deadline)
            resolve({ ok: true, plugin_name: 'demo', plugins: [] } as T)
          }, 45_000)
        })
    )

    const install =
      action === 'install'
        ? installAgentPlugin(request as never, { identifier: 'demo', profile: 'research' })
        : updateAgentPlugin(request as never, 'demo', 'failed', 'research')

    await vi.advanceTimersByTimeAsync(45_000)

    expect(await install).toMatchObject(action === 'install' ? { ok: true, pluginName: 'demo' } : { kind: 'applied' })
    expect(request).toHaveBeenCalledWith(
      'plugins.manage',
      expect.objectContaining({ action, profile: 'research' }),
      expect.any(Number)
    )
  })

  it.each(['install', 'update'])('marks a client timeout as an unknown %s outcome', async action => {
    const request = vi.fn(async () => {
      throw new Error('request timed out after 120s: plugins.manage')
    })

    const result =
      action === 'install'
        ? await installAgentPlugin(request as never, { identifier: 'demo' })
        : await updateAgentPlugin(request as never, 'demo', 'failed')

    expect(result).toMatchObject(action === 'install' ? { ok: false, timedOut: true } : { kind: 'unknown' })
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

  // The backend's settings save is what announces `plugin.<id>.settings.changed`; a new API key alone
  // is the change a plugin's Desktop half most needs to hear about.
  it('still makes the settings call when only a secret changed', async () => {
    $agentPluginsProfile.set('workbot')
    const request = vi.fn(async () => ({ ok: true, plugin: row({ key: 'demo', source: 'user' }) }))

    await saveAgentPluginSettings(request as never, {
      failMessage: 'fail',
      key: 'demo',
      profile: 'workbot',
      secrets: { DEMO_API_KEY: 'sk-2' },
      values: {},
      writeSecret: vi.fn(async () => ({ ok: true }))
    })

    expect(request).toHaveBeenCalledWith('plugins.manage', { action: 'settings', key: 'demo', profile: 'workbot', values: {} })
    expect(JSON.stringify(request.mock.calls)).not.toContain('sk-2')
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

import { afterEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestLocalMode } from '@/api/client'
import { $connection } from '@/store/session'

const { scopedGateway } = vi.hoisted(() => ({ scopedGateway: vi.fn() }))
vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestGatewayForAgent: (...args: unknown[]) => scopedGateway(...args)
}))

import {
  $agentPluginBusy,
  $agentPlugins,
  $agentPluginsOwner,
  agentPluginConnectionOwner,
  type AgentPluginRow,
  installAgentPlugin,
  isDesktopRelevantPlugin,
  loadAgentPlugins,
  normalizeAgentPluginRow,
  removeAgentPlugin,
  saveAgentPluginSettings,
  scopedAgentPluginRequest,
  toggleAgentPlugin,
  updateAgentPlugin
} from './agent-plugins'

const row = (partial: Partial<AgentPluginRow>): AgentPluginRow =>
  ({ name: partial.key ?? 'x', status: 'enabled', ...partial }) as AgentPluginRow

afterEach(() => {
  vi.useRealTimers()
  $agentPluginsOwner.set(null)
  $connection.set(null)
  setApiRequestConnection(null)
  setApiRequestLocalMode(false)
})

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

describe('transport-owned plugin inventory', () => {
  it('finishes immutable A settings after switching to B without changing B rows or busy state', async () => {
    const aScope = { connectionId: 'pinned-A', profile: 'default' }
    const bScope = { connectionId: 'pinned-B', profile: 'default' }
    const ambient = vi.fn()
    const a = scopedAgentPluginRequest(aScope, ambient as never)
    const b = scopedAgentPluginRequest(bScope, ambient as never)
    let finishB!: (value: unknown) => void
    scopedGateway.mockImplementation(
      async (connectionId: string, _profile: string, _method: string, params: Record<string, unknown>) => {
        if (params.action === 'list') {
          return { plugins: [row({ key: connectionId })] }
        }

        if (connectionId === 'pinned-B') {
          return new Promise(resolve => {
            finishB = resolve
          })
        }

        return { ok: true, plugin: row({ key: 'pinned-A', status: 'disabled' }) }
      }
    )
    await loadAgentPlugins(a, 'default')
    let finishFirst!: () => void
    const credentialWrite = vi.fn()

    const writeSecret = vi.fn(async (env: string, value: string) => {
      credentialWrite(env, value, aScope)

      if (env === 'FIRST_TEST_KEY') {
        await new Promise<void>(resolve => {
          finishFirst = resolve
        })
      }
    })

    const save = saveAgentPluginSettings(a, {
      key: 'pinned-A',
      profile: 'default',
      values: { retries: 2 },
      secrets: { FIRST_TEST_KEY: 'synthetic-first', SECOND_TEST_KEY: 'synthetic-second' },
      writeSecret,
      failMessage: 'failed'
    })

    expect(writeSecret).toHaveBeenCalledTimes(1)
    await loadAgentPlugins(b, 'default')
    const bRows = $agentPlugins.get()
    const bOwner = $agentPluginsOwner.get()
    const bMutation = toggleAgentPlugin(b, 'pinned-B', false, 'failed', 'default')
    finishFirst()
    expect(await save).toBe(true)
    expect(credentialWrite.mock.calls.map(call => call[2])).toEqual([aScope, aScope])
    expect(scopedGateway).toHaveBeenCalledWith(
      'pinned-A',
      'default',
      'plugins.manage',
      { action: 'settings', key: 'pinned-A', profile: 'default', values: { retries: 2 } },
      undefined,
      undefined,
      { spawnPriority: 'foreground' }
    )
    expect($agentPlugins.get()).toBe(bRows)
    expect($agentPluginsOwner.get()).toBe(bOwner)
    expect($agentPluginBusy.get()).toBe('pinned-B')
    expect(ambient).not.toHaveBeenCalled()
    finishB({ ok: true, plugin: row({ key: 'pinned-B', status: 'disabled' }) })
    expect(await bMutation).toBe(true)
  })

  function legacy(url: string) {
    setApiRequestConnection(null)
    setApiRequestLocalMode(false)
    $connection.set({ mode: 'remote', baseUrl: url } as never)
  }

  it('separates same-profile legacy endpoints on one host and rejects an old origin before RPC', async () => {
    legacy('https://placeholder-user:placeholder-pass@gateway.example/proxy-a/?ignored=1#fragment')
    const first = vi.fn(async () => ({ plugins: [row({ name: 'A' })] }))
    const a = scopedAgentPluginRequest('default', first as never)
    await loadAgentPlugins(a, 'default')
    expect(agentPluginConnectionOwner()).toBe('legacy:https://gateway.example/proxy-a')
    expect($agentPluginsOwner.get()).not.toContain('placeholder')
    expect($agentPluginsOwner.get()).not.toContain('ignored')
    legacy('https://gateway.example/proxy-b')
    const second = vi.fn(async () => ({ plugins: [row({ name: 'B' })] }))
    const b = scopedAgentPluginRequest('default', second as never)
    await loadAgentPlugins(b, 'default')
    expect($agentPlugins.get().map(plugin => plugin.name)).toEqual(['B'])
    expect(await removeAgentPlugin(a, 'A', 'failed', 'default')).toBe(false)
    expect(first).toHaveBeenCalledTimes(1)
  })

  it('stops later credential writes and config RPC when a legacy source changes during the first write', async () => {
    legacy('https://gateway.example/proxy-a')
    const ambient = vi.fn(async () => ({ plugins: [row({ name: 'shared', key: 'shared' })] }))
    const request = scopedAgentPluginRequest('default', ambient as never)
    await loadAgentPlugins(request, 'default')
    let finishFirst!: () => void

    const writeSecret = vi.fn(
      () =>
        new Promise<void>(resolve => {
          finishFirst = resolve
        })
    )

    const save = saveAgentPluginSettings(request, {
      key: 'shared',
      profile: 'default',
      values: { retries: 2 },
      secrets: { FIRST_TEST_KEY: 'synthetic-first', SECOND_TEST_KEY: 'synthetic-second' },
      writeSecret,
      failMessage: 'failed'
    })

    expect(writeSecret).toHaveBeenCalledTimes(1)
    legacy('https://gateway.example/proxy-b')
    finishFirst()
    expect(await save).toBe(false)
    expect(writeSecret).toHaveBeenCalledTimes(1)
    expect(ambient).not.toHaveBeenCalledWith('plugins.manage', expect.objectContaining({ action: 'settings' }))
  })

  it('revokes later credential writes as soon as canonical registry ownership changes before its descriptor', async () => {
    $connection.set({ mode: 'remote', connectionId: 'source-A', baseUrl: 'https://gateway.example/proxy-a' } as never)
    setApiRequestConnection('source-A')
    const ambient = vi.fn(async () => ({ plugins: [row({ name: 'shared', key: 'shared' })] }))
    const request = scopedAgentPluginRequest('default', ambient as never)
    await loadAgentPlugins(request, 'default')
    let finishFirst!: () => void

    const writeSecret = vi.fn(
      () =>
        new Promise<void>(resolve => {
          finishFirst = resolve
        })
    )

    const save = saveAgentPluginSettings(request, {
      key: 'shared',
      profile: 'default',
      values: { retries: 2 },
      secrets: { FIRST_TEST_KEY: 'synthetic-first', SECOND_TEST_KEY: 'synthetic-second' },
      writeSecret,
      failMessage: 'failed'
    })

    setApiRequestConnection('source-B')
    expect($connection.get()?.connectionId).toBe('source-A')
    finishFirst()
    expect(await save).toBe(false)
    expect(writeSecret).toHaveBeenCalledTimes(1)
    expect(ambient).not.toHaveBeenCalledWith('plugins.manage', expect.objectContaining({ action: 'settings' }))
  })

  it('revokes old and new bare bindings when canonical ownership clears but a registered descriptor remains', async () => {
    $connection.set({ mode: 'remote', connectionId: 'source-A', baseUrl: 'https://gateway.example/proxy-a' } as never)
    setApiRequestConnection('source-A')
    const ambient = vi.fn(async () => ({ plugins: [row({ name: 'shared', key: 'shared' })] }))
    const old = scopedAgentPluginRequest('default', ambient as never)
    await loadAgentPlugins(old, 'default')
    let finishFirst!: () => void

    const writeSecret = vi.fn(
      () =>
        new Promise<void>(resolve => {
          finishFirst = resolve
        })
    )

    const save = saveAgentPluginSettings(old, {
      key: 'shared',
      profile: 'default',
      values: { retries: 2 },
      secrets: { FIRST_TEST_KEY: 'synthetic-first', SECOND_TEST_KEY: 'synthetic-second' },
      writeSecret,
      failMessage: 'failed'
    })

    setApiRequestConnection(null)
    expect($connection.get()?.connectionId).toBe('source-A')
    expect(agentPluginConnectionOwner()).toBeUndefined()
    finishFirst()
    expect(await save).toBe(false)
    expect(writeSecret).toHaveBeenCalledTimes(1)
    const fresh = scopedAgentPluginRequest('default', ambient as never)
    expect(await removeAgentPlugin(fresh, 'shared', 'failed', 'default')).toBe(false)
    expect(ambient).toHaveBeenCalledTimes(1)
    setApiRequestConnection('source-A')
    expect(await removeAgentPlugin(old, 'shared', 'failed', 'default')).toBe(false)
    expect(ambient).toHaveBeenCalledTimes(1)
  })
  it('does not deduplicate same-named profiles from different transports or accept old A after A → B → A', async () => {
    let resolveA!: (result: { plugins: AgentPluginRow[] }) => void
    let resolveB!: (result: { plugins: AgentPluginRow[] }) => void

    const a = vi
      .fn()
      .mockImplementationOnce(
        () =>
          new Promise(resolve => {
            resolveA = resolve
          })
      )
      .mockResolvedValue({ plugins: [row({ name: 'new-A' })] })

    const b = vi.fn(
      () =>
        new Promise(resolve => {
          resolveB = resolve
        })
    )

    const oldA = loadAgentPlugins(a as never, 'default')
    expect(loadAgentPlugins(a as never, 'default')).toBe(oldA)
    const pendingB = loadAgentPlugins(b as never, 'default')
    expect(b).toHaveBeenCalledTimes(1)
    await loadAgentPlugins(a as never, 'default')
    resolveB({ plugins: [row({ name: 'B' })] })
    resolveA({ plugins: [row({ name: 'old-A' })] })
    await Promise.all([oldA, pendingB])
    expect($agentPlugins.get().map(plugin => plugin.name)).toEqual(['new-A'])
  })

  it.each(['toggle', 'remove', 'settings', 'update'] as const)(
    'ignores late %s completion after A → B → A',
    async action => {
      let complete!: (result: unknown) => void
      const current = row({ name: 'shared', key: 'shared', settings_schema: [] })

      const a = vi.fn(async (_method: string, params: Record<string, unknown>) =>
        params.action === 'list'
          ? { plugins: [current] }
          : await new Promise(resolve => {
              complete = resolve
            })
      )

      const b = vi.fn(async () => ({ plugins: [row({ name: 'B' })] }))
      await loadAgentPlugins(a as never, 'default')

      const mutation =
        action === 'toggle'
          ? toggleAgentPlugin(a as never, 'shared', false, 'failed', 'default')
          : action === 'remove'
            ? removeAgentPlugin(a as never, 'shared', 'failed', 'default')
            : action === 'update'
              ? updateAgentPlugin(a as never, 'shared', 'failed', 'default')
              : saveAgentPluginSettings(a as never, {
                  key: 'shared',
                  values: { retries: 9 },
                  secrets: {},
                  writeSecret: vi.fn(),
                  failMessage: 'failed',
                  profile: 'default'
                })

      await loadAgentPlugins(b as never, 'default')
      await loadAgentPlugins(a as never, 'default')
      const owner = $agentPluginsOwner.get()
      const listCalls = a.mock.calls.filter(([, params]) => params.action === 'list').length
      complete({
        ok: true,
        plugin: row({ name: 'shared', key: 'shared', status: 'disabled', settings_schema: [{ key: 'wrong' }] as never })
      })
      await mutation
      expect($agentPlugins.get()).toEqual([normalizeAgentPluginRow(current)])
      expect($agentPluginsOwner.get()).toBe(owner)
      expect($agentPluginBusy.get()).toBeNull()
      expect(a.mock.calls.filter(([, params]) => params.action === 'list')).toHaveLength(listCalls)
    }
  )

  it('revokes A rows immediately while same-profile B is pending and after B fails', async () => {
    const a = vi.fn(async () => ({ plugins: [row({ name: 'A-only' })] }))
    let rejectB!: (error: Error) => void

    const b = vi.fn(
      () =>
        new Promise((_resolve, reject) => {
          rejectB = reject
        })
    )

    await loadAgentPlugins(a as never, 'default')
    const pending = loadAgentPlugins(b as never, 'default')
    expect($agentPlugins.get()).toEqual([])
    rejectB(new Error('B unavailable'))
    await pending
    expect($agentPlugins.get()).toEqual([])
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

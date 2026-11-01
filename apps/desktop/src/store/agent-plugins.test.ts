import { JsonRpcGatewayError } from '@hermes/shared'
import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  $agentPlugins,
  $agentPluginsByOwner,
  $agentPluginsError,
  $agentPluginsErrorByOwner,
  $agentPluginsLoadingByOwner,
  $agentPluginsProfile,
  $agentPluginsStatus,
  type AgentPluginRow,
  inspectAgentPlugin,
  installAgentPlugin,
  isDesktopRelevantPlugin,
  loadAgentPlugins,
  normalizeAgentPluginRow,
  pluginInventoryKey,
  saveAgentPluginSettings,
  toggleAgentPlugin,
  updateAgentPlugin
} from './agent-plugins'

const row = (partial: Partial<AgentPluginRow>): AgentPluginRow =>
  ({ name: partial.key ?? 'x', status: 'enabled', ...partial }) as AgentPluginRow

afterEach(() => {
  vi.useRealTimers()
  $agentPlugins.set([])
  $agentPluginsStatus.set('idle')
  $agentPluginsError.set(null)
  $agentPluginsProfile.set(undefined)
  $agentPluginsByOwner.set({})
  $agentPluginsErrorByOwner.set({})
  $agentPluginsLoadingByOwner.set({})
})

describe('loadAgentPlugins connection ownership', () => {
  it('does not coalesce same-named profiles across backends or publish either into the shared list', async () => {
    let finishCloud!: (value: { plugins: AgentPluginRow[] }) => void
    const cloud = vi.fn(() => new Promise<{ plugins: AgentPluginRow[] }>(resolve => { finishCloud = resolve }))
    const local = vi.fn(async () => ({ plugins: [row({ key: 'local-plugin' })] }))
    const slow = loadAgentPlugins(cloud as never, 'workbot', 'cloud')
    const fast = loadAgentPlugins(local as never, 'workbot', 'local')
    await fast
    finishCloud({ plugins: [row({ key: 'cloud-plugin' })] })
    await slow
    expect(cloud).toHaveBeenCalledTimes(1)
    expect(local).toHaveBeenCalledTimes(1)
    expect($agentPlugins.get()).toEqual([])
    expect($agentPluginsProfile.get()).toBeUndefined()
    expect($agentPluginsByOwner.get()[pluginInventoryKey('cloud', 'workbot')].map(item => item.key))
      .toEqual(['cloud-plugin'])
    expect($agentPluginsByOwner.get()[pluginInventoryKey('local', 'workbot')].map(item => item.key))
      .toEqual(['local-plugin'])
  })

  it('does not let a pre-toggle list response undo a completed owner mutation', async () => {
    let finish!: (value: { plugins: AgentPluginRow[] }) => void
    const original = row({ key: 'weather', status: 'disabled' })
    // The ambient backend's list for the same label is another backend's state.
    $agentPlugins.set([original])
    $agentPluginsProfile.set('workbot')
    $agentPluginsByOwner.set({ [pluginInventoryKey('local', 'workbot')]: [original] })
    const list = loadAgentPlugins(vi.fn(() => new Promise(resolve => { finish = resolve })) as never, 'workbot', 'local')
    const updated = row({ key: 'weather', status: 'enabled' })

    const ok = await toggleAgentPlugin(vi.fn(async () => ({ ok: true, plugin: updated })) as never,
      'weather', true, 'failed', 'workbot', 'local')

    expect(ok).toBe(true)
    finish({ plugins: [original] })
    await list
    expect($agentPluginsByOwner.get()[pluginInventoryKey('local', 'workbot')][0].status).toBe('enabled')
    expect($agentPlugins.get()[0].status).toBe('disabled')
  })

  it('starts a fresh owner list after install while an older pre-install list is in flight', async () => {
    let finishOld!: (value: { plugins: AgentPluginRow[] }) => void
    const installed = row({ key: 'weather', catalog_name: 'weather' })

    const request = vi.fn(() => request.mock.calls.length === 1
      ? new Promise<{ plugins: AgentPluginRow[] }>(resolve => { finishOld = resolve })
      : Promise.resolve({ plugins: [installed] }))

    const stale = loadAgentPlugins(request as never, 'workbot', 'local')
    const fresh = loadAgentPlugins(request as never, 'workbot', 'local', { force: true })
    expect(request).toHaveBeenCalledTimes(2)
    await fresh
    finishOld({ plugins: [] })
    await stale
    expect($agentPluginsByOwner.get()[pluginInventoryKey('local', 'workbot')].map(item => item.catalog_name))
      .toEqual(['weather'])
  })

  it('leaves the shared list, its label and its status to the ambient backend', async () => {
    let finishAmbient!: (value: { plugins: AgentPluginRow[] }) => void

    await loadAgentPlugins(vi.fn(async () => ({ plugins: [row({ key: 'ambient' })] })) as never, 'default')

    const ambient = loadAgentPlugins(
      vi.fn(() => new Promise<{ plugins: AgentPluginRow[] }>(resolve => { finishAmbient = resolve })) as never,
      'default',
      null,
      { force: true }
    )

    // Settings reads the shared list by profile label: the same label on another backend must not land there.
    await loadAgentPlugins(vi.fn(async () => ({ plugins: [row({ key: 'remote' })] })) as never, 'default', 'cloud')
    expect($agentPlugins.get().map(item => item.key)).toEqual(['ambient'])
    await loadAgentPlugins(vi.fn(async () => { throw new Error('cloud is unreachable') }) as never, 'default', 'cloud', {
      force: true
    })
    finishAmbient({ plugins: [row({ key: 'ambient-2' })] })
    await ambient
    expect($agentPlugins.get().map(item => item.key)).toEqual(['ambient-2'])
    expect($agentPluginsProfile.get()).toBe('default')
    expect($agentPluginsStatus.get()).toBe('ready')
    expect($agentPluginsError.get()).toBeNull()
    expect($agentPluginsErrorByOwner.get()[pluginInventoryKey('cloud', 'default')]).toBe('cloud is unreachable')
  })

  it('retains owner A list failure even when owner B refreshes first', async () => {
    let failA!: (reason: Error) => void

    const a = loadAgentPlugins(vi.fn(() => new Promise((_resolve, reject) => { failA = reject })) as never,
      'workbot', 'cloud')

    await loadAgentPlugins(vi.fn(async () => ({ plugins: [] })) as never, 'workbot', 'local')
    failA(new Error('cloud is unreachable'))
    await a
    expect($agentPluginsErrorByOwner.get()[pluginInventoryKey('cloud', 'workbot')]).toBe('cloud is unreachable')
    expect($agentPluginsErrorByOwner.get()[pluginInventoryKey('local', 'workbot')]).toBeNull()
  })

  it('refreshes the same owner cache after update rather than stamping a shared null owner', async () => {
    const request = vi.fn(async (_method: string, params: Record<string, unknown>) =>
      params.action === 'update' ? { ok: true } : { plugins: [row({ key: 'weather', version: '2' })] })

    $agentPluginsByOwner.set({ [pluginInventoryKey('local', 'workbot')]: [row({ key: 'weather', version: '1' })] })
    const result = await updateAgentPlugin(request as never, 'weather', 'failed', 'workbot', {}, 'local')
    expect(result.kind).toBe('applied')
    expect($agentPluginsProfile.get()).toBeUndefined()
    expect($agentPluginsByOwner.get()[pluginInventoryKey('local', 'workbot')][0].version).toBe('2')
  })
})

describe('installAgentPlugin', () => {
  it('preserves typed app refusal on an install race without mistaking it for a timeout', async () => {
    const guidance = {
      reason: 'version_too_old',
      app_title: 'Godot 4',
      required_version: '4.3',
      found_version: '4.2'
    }

    const request = vi.fn(async () => {
      throw new JsonRpcGatewayError('app unavailable', { code: 5026, data: guidance })
    })

    expect(
      await installAgentPlugin(request as never, { identifier: 'game-editor', catalogName: 'game-editor' })
    ).toMatchObject({
      ok: false,
      timedOut: false,
      appGuidance: guidance
    })
  })

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

  const reviewed = 'e404b6816b20237a82f6fbe2749b5a576cf75dba'

  it('never resends without the inspected commit on its own when a backend predates inspected_sha', async () => {
    // The admission refusal an origin/main backend gives the field (validate_params, handler never ran).
    // Installing there without it takes whatever the branch holds now: only the user may choose that.
    const request = vi.fn(async (_method: string, params: Record<string, unknown>) => {
      if ('inspected_sha' in params) {
        throw new JsonRpcGatewayError(
          'invalid params for plugins.manage: inspected_sha: Extra inputs are not permitted — the client and the ' +
            'Hermes backend are out of sync (different versions); run `hermes update` and restart both',
          { code: 4000 }
        )
      }

      return { ok: true, plugin_name: 'unreviewed-tip', enabled: true }
    })

    expect(
      await installAgentPlugin(request as never, {
        identifier: 'https://github.com/example/pkg',
        inspectedSha: reviewed,
        profile: 'research'
      })
    ).toMatchObject({ ok: false, inspectedShaUnsupported: true })
    expect(request).toHaveBeenCalledTimes(1)

    // The user's explicit "Install anyway" is an ordinary legacy install, still on the originating profile.
    expect(
      await installAgentPlugin(request as never, { identifier: 'https://github.com/example/pkg', profile: 'research' })
    ).toMatchObject({ ok: true, pluginName: 'unreviewed-tip' })
    expect(request.mock.calls[1][1]).toMatchObject({ action: 'install', profile: 'research' })
    expect(request.mock.calls[1][1]).not.toHaveProperty('inspected_sha')
  })

  it('never drops the inspected commit for an install failure that merely quotes the admission refusal', async () => {
    // A git server controls its clone error's text; the backend forwards it as a 5026 install failure.
    // Only a code-4000 refusal OPENING with the admission prefix means "this backend predates the field".
    const quoted =
      'Git clone failed: remote: invalid params for plugins.manage: inspected_sha: Extra inputs are not permitted'

    for (const error of [
      new JsonRpcGatewayError('invalid params for plugins.manage: inspected_sha: Extra inputs are not permitted', {
        code: 5026
      }),
      new JsonRpcGatewayError(quoted, { code: 4000 }),
      new Error('invalid params for plugins.manage: inspected_sha: Extra inputs are not permitted')
    ]) {
      const request = vi.fn(async () => {
        throw error
      })

      expect(
        await installAgentPlugin(request as never, {
          identifier: 'https://github.com/example/pkg',
          inspectedSha: reviewed
        })
      ).toMatchObject({ ok: false, inspectedShaUnsupported: false })
      expect(request).toHaveBeenCalledTimes(1)
    }
  })

  it('reports a refusal carrying the backend source-moved marker as sourceMoved, and only that one', async () => {
    const refusing = (data?: unknown) =>
      vi.fn(async () => {
        throw new JsonRpcGatewayError('The source changed since you reviewed it.', { code: 5026, data })
      })

    const install = (request: ReturnType<typeof refusing>) =>
      installAgentPlugin(request as never, { identifier: 'https://github.com/example/pkg', inspectedSha: reviewed })

    expect(await install(refusing({ reason: 'source_moved' }))).toMatchObject({ ok: false, sourceMoved: true })
    expect(await install(refusing())).toMatchObject({ ok: false, sourceMoved: false })
  })

  it('holds a current backend to the inspected commit and never resends a refusal of the moved branch', async () => {
    const request = vi.fn(async () => {
      throw new JsonRpcGatewayError(
        'The source changed since you reviewed it: it is now at commit 0a415bc5c1d2, not e404b6816b20.',
        { code: 5026 }
      )
    })

    expect(
      await installAgentPlugin(request as never, {
        identifier: 'https://github.com/example/pkg',
        inspectedSha: reviewed
      })
    ).toMatchObject({ ok: false })
    expect(request).toHaveBeenCalledTimes(1)
    expect(request.mock.calls[0]).toEqual([
      'plugins.manage',
      expect.objectContaining({ inspected_sha: reviewed }),
      120_000
    ])
  })
})

describe('inspectAgentPlugin', () => {
  const answering = (sha: unknown) =>
    vi.fn(async () => ({ ok: true, agent: true, agent_name: 'demo', desktop: false, warnings: [], sha }))

  it('carries the commit the backend read, lower-cased, so the install can be held to it', async () => {
    const sha = 'ABCDEF0123456789abcdef0123456789abcdef01'
    const request = answering(sha)

    expect(await inspectAgentPlugin(request as never, { identifier: 'owner/repo', profile: 'research' })).toMatchObject(
      {
        ok: true,
        agentName: 'demo',
        sha: sha.toLowerCase()
      }
    )
    expect(request).toHaveBeenCalledWith(
      'plugins.manage',
      { action: 'inspect', identifier: 'owner/repo', profile: 'research' },
      expect.any(Number)
    )
  })

  it('drops a commit that is not a full SHA', async () => {
    for (const sha of ['main', 'abcdef0', null, undefined]) {
      expect(await inspectAgentPlugin(answering(sha) as never, { identifier: 'owner/repo' })).not.toHaveProperty('sha')
    }
  })

  it('marks a backend without the inspect action unavailable, and a transport failure retryable', async () => {
    const refusing = vi.fn(async () => {
      throw new JsonRpcGatewayError('unknown plugins action: inspect', { code: 4017 })
    })

    const timingOut = vi.fn(async () => {
      throw new Error('request timed out after 120s: plugins.manage')
    })

    expect(await inspectAgentPlugin(refusing as never, { identifier: 'owner/repo' })).toMatchObject({
      ok: false,
      unavailable: true
    })
    expect(await inspectAgentPlugin(timingOut as never, { identifier: 'owner/repo' })).toMatchObject({
      ok: false,
      unavailable: false
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

    expect(request).toHaveBeenCalledWith('plugins.manage', {
      action: 'settings',
      key: 'demo',
      profile: 'workbot',
      values: {}
    })
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

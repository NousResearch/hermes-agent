import { afterEach, describe, expect, it, vi } from 'vitest'

import { $pluginRecords } from '@/contrib/plugins-store'
import { discoverRuntimePlugins } from '@/contrib/runtime-loader'
import { $notifications } from '@/store/notifications'

import { $pluginUpdateAllRunning, isPluginUpdateAllRunning, updateAllAgentPlugins } from './plugin-update-all'

vi.mock('@/contrib/runtime-loader', () => ({ discoverRuntimePlugins: vi.fn().mockResolvedValue(undefined) }))

describe('plugin update-all running state', () => {
  it('refreshes each updated remote desktop half and counts partial failures', async () => {
    const install = vi.fn().mockResolvedValueOnce({ ok: true, pluginName: 'one' }).mockResolvedValueOnce({ ok: false })
    vi.mocked(discoverRuntimePlugins).mockImplementationOnce(async () => {
      $pluginRecords.set({ one: { id: 'one', name: 'one', kind: 'disk', status: 'loaded', packageName: 'one', packageOrigin: { sha: 'b'.repeat(40) } } })
    })
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { installDesktopPlugin: install } })
    const rows = ['one', 'two'].map(name => ({ name, status: 'updated', has_desktop_half: true, installed_repo: `https://example.com/${name}`, installed_sha: 'b'.repeat(40) }))
    const request = vi.fn().mockImplementation(async (_method, params) => params.action === 'update_all' ? { results: rows } : { plugins: [] })

    const summary = await updateAllAgentPlugins(request, 'work', true)
    expect(install).toHaveBeenCalledTimes(2)
    expect(summary.updated).toBe(1)
    expect(summary.failed).toBe(1)
    expect(summary.rows[1].detail).toContain('two')
  })
  afterEach(() => {
    $pluginUpdateAllRunning.set(new Set())
    $notifications.set([])
    $pluginRecords.set({})
    delete document.documentElement.dataset.hermesDesktopHost
    vi.resetAllMocks()
    vi.mocked(discoverRuntimePlugins).mockResolvedValue(undefined)
  })

  it.each(['missing', 'error', 'mismatch'])('counts resolved discovery %s as a partial failure', async scenario => {
    const sha = 'c'.repeat(40)
    const install = vi.fn().mockResolvedValue({ ok: true, pluginName: 'one' })
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { installDesktopPlugin: install } })
    vi.mocked(discoverRuntimePlugins).mockImplementationOnce(async () => {
      if (scenario !== 'missing') {
        $pluginRecords.set({ one: { id: 'one', name: 'one', kind: 'disk', status: scenario === 'error' ? 'error' : 'loaded', packageName: 'one', packageOrigin: { sha: scenario === 'mismatch' ? 'd'.repeat(40) : sha } } })
      }
    })

    const request = vi.fn().mockImplementation(async (_method, params) => params.action === 'update_all'
      ? { results: [{ name: 'one', status: 'updated', has_desktop_half: true, installed_repo: 'repo', installed_sha: sha }] } : { plugins: [] })

    const summary = await updateAllAgentPlugins(request, 'work', true)

    expect(summary).toMatchObject({ updated: 0, failed: 1 })
    expect($notifications.get().some(n => n.kind === 'warning' && n.message.includes('one'))).toBe(true)
  })

  it('uses the scoped old-backend list only as a flag and warns without a fresh pin', async () => {
    const install = vi.fn()
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { installDesktopPlugin: install } })

    const request = vi.fn().mockImplementation(async (_method, params) => params.action === 'update_all'
      ? { results: [{ name: 'old', status: 'updated' }] }
      : { plugins: [{ name: 'old', has_desktop_half: true, installed_sha: 'a'.repeat(40) }] })

    const summary = await updateAllAgentPlugins(request, 'other', true)
    expect(request).toHaveBeenCalledWith('plugins.manage', { action: 'list', profile: 'other' })
    expect(install).not.toHaveBeenCalled()
    expect(summary).toMatchObject({ updated: 0, failed: 1 })
    expect($notifications.get().some(n => n.kind === 'warning' && n.message.includes('old'))).toBe(true)
  })

  it('uses each fresh row for multiple successes and skips other statuses', async () => {
    const install = vi.fn().mockResolvedValue({ ok: true, pluginName: 'desktop' })
    vi.mocked(discoverRuntimePlugins).mockImplementation(async () => {
      $pluginRecords.set({ desktop: { id: 'desktop', name: 'desktop', kind: 'disk', status: 'loaded', packageName: 'desktop', packageOrigin: { sha: 'd'.repeat(40) } } })
    })
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { installDesktopPlugin: install } })
    const sha = 'd'.repeat(40)

    const rows = [
      { name: 'catalog', status: 'updated', has_desktop_half: true, installed_repo: 'repo-one', installed_subdir: 'sub', installed_sha: sha },
      { name: 'git', status: 'updated', has_desktop_half: true, installed_repo: 'repo-two', installed_sha: sha },
      { name: 'plain', status: 'updated', has_desktop_half: false },
      { name: 'current', status: 'current', has_desktop_half: true },
      { name: 'failed', status: 'failed', has_desktop_half: true },
      { name: 'review', status: 'needs_review', has_desktop_half: true }
    ]

    const request = vi.fn().mockImplementation(async (_method, params) => params.action === 'update_all'
      ? { results: rows } : { plugins: [{ name: 'catalog', catalog_name: 'catalog-id', has_desktop_half: true }] })

    const summary = await updateAllAgentPlugins(request, 'work', true)

    expect(install).toHaveBeenCalledTimes(2)
    expect(install).toHaveBeenNthCalledWith(1, { identifier: 'repo-one#sub', force: true, sha, catalogName: 'catalog-id' })
    expect(install).toHaveBeenNthCalledWith(2, { identifier: 'repo-two', force: true, sha, catalogName: undefined })
    expect(summary).toMatchObject({ updated: 3, failed: 1, review: 1 })
  })

  it.each(['local', 'browser'])('skips cloning in %s mode', async mode => {
    const install = vi.fn()
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { installDesktopPlugin: install } })

    if (mode === 'browser') {
      document.documentElement.dataset.hermesDesktopHost = 'browser'
    }

    const request = vi.fn().mockImplementation(async (_method, params) => params.action === 'update_all'
      ? { results: [{ name: 'one', status: 'updated', has_desktop_half: true }] } : { plugins: [] })

    expect((await updateAllAgentPlugins(request, null, mode === 'browser')).updated).toBe(1)
    expect(install).not.toHaveBeenCalled()
  })

  it('marks only the profile whose run is in flight', async () => {
    let finish!: (value: unknown) => void
    const request = vi.fn(() => new Promise(resolve => (finish = resolve))) as never
    const run = updateAllAgentPlugins(request, 'work')

    expect(isPluginUpdateAllRunning($pluginUpdateAllRunning.get(), 'work')).toBe(true)
    expect(isPluginUpdateAllRunning($pluginUpdateAllRunning.get(), null)).toBe(false)
    expect(isPluginUpdateAllRunning($pluginUpdateAllRunning.get(), 'home')).toBe(false)

    finish({ results: [] })
    await run
    expect(isPluginUpdateAllRunning($pluginUpdateAllRunning.get(), 'work')).toBe(false)
  })
})

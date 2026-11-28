import { afterEach, describe, expect, it, vi } from 'vitest'

import { $pluginRecords } from '@/contrib/plugins-store'
import { discoverRuntimePlugins } from '@/contrib/runtime-loader'
import { en } from '@/i18n/en'
import { $confirmRequest, settleConfirm } from '@/store/confirm'
import { $notifications } from '@/store/notifications'

import { runPluginUpdate } from './plugin-update'

vi.mock('@/contrib/runtime-loader', () => ({ discoverRuntimePlugins: vi.fn().mockResolvedValue(undefined) }))

const p = en.skills.plugins
const tern = { catalog_name: undefined, name: 'hermes-tern' }

const caution = {
  ok: false,
  scan_blocked: true,
  scan_verdict: 'caution',
  scan_findings: [{ category: 'execution', description: 'eval', file: 'helper.py', line: 1, severity: 'high' }]
}

afterEach(() => {
  settleConfirm(false)
  delete document.documentElement.dataset.hermesDesktopHost
  vi.clearAllMocks()
  $notifications.set([])
  $pluginRecords.set({})
})

describe('runPluginUpdate scan override', () => {
  it('refreshes a remote desktop half at the newly installed commit', async () => {
    const sha = 'a'.repeat(40)
    const install = vi.fn().mockResolvedValue({ ok: true, pluginName: 'tern' })
    vi.mocked(discoverRuntimePlugins).mockImplementationOnce(async () => {
      $pluginRecords.set({ tern: { id: 'tern', name: 'tern', kind: 'disk', status: 'loaded', packageName: 'tern', packageOrigin: { sha: 'a'.repeat(40) } } })
    })
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { installDesktopPlugin: install } })
    const request = vi.fn().mockResolvedValueOnce({ ok: true, has_desktop_half: true, installed_repo: 'https://example.com/repo.git', installed_subdir: 'pkg', installed_sha: sha }).mockResolvedValue({ plugins: [] })

    expect((await runPluginUpdate(request, { ...tern, has_desktop_half: true }, null, p, true)).kind).toBe('applied')
    expect($notifications.get().some(n => n.kind === 'success')).toBe(true)
    expect(install).toHaveBeenCalledWith({ identifier: 'https://example.com/repo.git#pkg', force: true, sha, catalogName: undefined })
    expect(vi.mocked(discoverRuntimePlugins).mock.invocationCallOrder[0]).toBeGreaterThan(install.mock.invocationCallOrder[0])
  })
  it('warns when discovery resolves without the installed plugin', async () => {
    const install = vi.fn().mockResolvedValue({ ok: true, pluginName: 'tern' })
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { installDesktopPlugin: install } })
    const request = vi.fn().mockResolvedValueOnce({ ok: true, has_desktop_half: true, installed_repo: 'repo', installed_sha: 'a'.repeat(40) }).mockResolvedValue({ plugins: [] })

    expect((await runPluginUpdate(request, tern, null, p, true)).kind).toBe('partial')
    expect($notifications.get().some(n => n.kind === 'warning')).toBe(true)
  })
  it.each(['error', 'mismatch', 'record error', 'wrong name'])('warns for discovered %s', async scenario => {
    const sha = 'a'.repeat(40)
    const install = vi.fn().mockResolvedValue({ ok: true, pluginName: 'tern' })
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { installDesktopPlugin: install } })
    vi.mocked(discoverRuntimePlugins).mockImplementationOnce(async () => {
      $pluginRecords.set({ tern: { id: 'tern', name: 'tern', kind: 'disk', status: scenario === 'error' ? 'error' : 'loaded', error: scenario === 'record error' ? 'failed' : undefined, packageName: scenario === 'wrong name' ? 'other' : 'tern', packageOrigin: { sha: scenario === 'mismatch' ? 'b'.repeat(40) : sha } } })
    })
    const request = vi.fn().mockResolvedValueOnce({ ok: true, has_desktop_half: true, installed_repo: 'repo', installed_sha: sha }).mockResolvedValue({ plugins: [] })
    expect((await runPluginUpdate(request, tern, null, p, true)).kind).toBe('partial')
    expect($notifications.get().some(n => n.kind === 'warning')).toBe(true)
    expect($notifications.get().some(n => n.kind === 'success')).toBe(false)
  })
  it('accepts a disabled desktop half with a case-insensitive matching SHA', async () => {
    const sha = 'a'.repeat(40)
    const install = vi.fn().mockResolvedValue({ ok: true, pluginName: 'tern' })
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { installDesktopPlugin: install } })
    vi.mocked(discoverRuntimePlugins).mockImplementationOnce(async () => {
      $pluginRecords.set({ tern: { id: 'tern', name: 'tern', kind: 'disk', status: 'disabled', packageName: 'tern', packageOrigin: { sha: sha.toUpperCase() } } })
    })
    const request = vi.fn().mockResolvedValueOnce({ ok: true, has_desktop_half: true, installed_repo: 'repo', installed_sha: sha }).mockResolvedValue({ plugins: [] })
    expect((await runPluginUpdate(request, tern, null, p, true)).kind).toBe('applied')
  })
  it.each(['missing provenance', 'short sha', 'rejected', 'throws', 'discovery', 'missing bridge'])('warns on %s without a success toast', async failure => {
    const install = vi.fn().mockResolvedValue(failure === 'rejected' ? { ok: false } : { ok: true })

    if (failure === 'throws') {
      install.mockRejectedValue(new Error('install'))
    }

    if (failure === 'discovery') {
      vi.mocked(discoverRuntimePlugins).mockRejectedValueOnce(new Error('scan'))
    }

    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: failure === 'missing bridge' ? {} : { installDesktopPlugin: install } })
    const provenance = failure === 'missing provenance' ? {} : { installed_repo: 'https://example.com/repo', installed_sha: 'a'.repeat(failure === 'short sha' ? 8 : 40) }
    const request = vi.fn().mockResolvedValueOnce({ ok: true, ...provenance }).mockResolvedValue({ plugins: [] })

    expect((await runPluginUpdate(request, { ...tern, has_desktop_half: true }, null, p, true)).kind).toBe('partial')
    expect($notifications.get().some(n => n.kind === 'warning' && n.message.includes(tern.name))).toBe(true)
    expect($notifications.get().some(n => n.kind === 'success')).toBe(false)

    if (failure === 'missing provenance' || failure === 'short sha') {
      expect(install).not.toHaveBeenCalled()
    }
  })
  it.each(['local', 'browser', 'no half', 'unchanged', 'failed'])('does not clone for %s', async scenario => {
    const install = vi.fn().mockResolvedValue({ ok: true })
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { installDesktopPlugin: install } })

    if (scenario === 'browser') {
      document.documentElement.dataset.hermesDesktopHost = 'browser'
    }

    const reply = { ok: scenario !== 'failed', unchanged: scenario === 'unchanged', has_desktop_half: scenario !== 'no half', installed_repo: 'https://example.com/repo', installed_sha: 'a'.repeat(40) }
    const request = vi.fn().mockResolvedValueOnce(reply).mockResolvedValue({ plugins: [] })
    await runPluginUpdate(request, { ...tern, has_desktop_half: true }, null, p, scenario !== 'local')
    expect(install).not.toHaveBeenCalled()
  })
  it('warns if the updated package removed its previously installed desktop half', async () => {
    const install = vi.fn()
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { installDesktopPlugin: install } })
    const request = vi.fn().mockResolvedValueOnce({ ok: true, has_desktop_half: false }).mockResolvedValue({ plugins: [] })

    expect((await runPluginUpdate(request, { ...tern, has_desktop_half: true }, null, p, true)).kind).toBe('partial')
    expect(install).not.toHaveBeenCalled()
    expect($notifications.get().some(n => n.kind === 'success')).toBe(false)
  })
  it('uses the catalog name with the fresh commit', async () => {
    const install = vi.fn().mockResolvedValue({ ok: true, pluginName: 'tern' })
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { installDesktopPlugin: install } })
    vi.mocked(discoverRuntimePlugins).mockImplementationOnce(async () => {
      $pluginRecords.set({ tern: { id: 'tern', name: 'tern', kind: 'disk', status: 'loaded', packageName: 'tern', packageOrigin: { sha: 'c'.repeat(40) } } })
    })
    const request = vi.fn().mockResolvedValueOnce({ ok: true, has_desktop_half: true, installed_repo: 'repo', installed_sha: 'c'.repeat(40) }).mockResolvedValue({ plugins: [] })
    expect((await runPluginUpdate(request, { ...tern, catalog_name: 'catalog-tern' }, null, p, true)).kind).toBe('applied')
    expect(install).toHaveBeenCalledWith(expect.objectContaining({ catalogName: 'catalog-tern', sha: 'c'.repeat(40) }))
  })
  it('shows a caution finding, and only "Update anyway" retries the update with force', async () => {
    const request = vi.fn().mockResolvedValueOnce(caution).mockResolvedValueOnce({ ok: true, unchanged: false })

    const done = runPluginUpdate(request, tern, null, p)

    await vi.waitFor(() => expect($confirmRequest.get()?.title).toBe(p.updateScanTitle('hermes-tern')))
    expect($confirmRequest.get()?.description).toContain('HIGH  helper.py:1  eval')
    expect($confirmRequest.get()?.destructive).toBe(true)
    expect(request).toHaveBeenCalledTimes(1)

    settleConfirm(true)

    expect(await done).toEqual({ kind: 'applied' })
    expect(request).toHaveBeenNthCalledWith(
      2,
      'plugins.manage',
      expect.objectContaining({ action: 'update', force: true, name: 'hermes-tern' }),
      expect.any(Number)
    )
  })

  it('cancel leaves the plugin alone, and a dangerous verdict never offers the override', async () => {
    const cancelled = vi.fn().mockResolvedValue(caution)
    const done = runPluginUpdate(cancelled, tern, null, p)

    await vi.waitFor(() => expect($confirmRequest.get()).not.toBeNull())
    settleConfirm(false)
    expect((await done).kind).toBe('scan')
    expect(cancelled).toHaveBeenCalledTimes(1)

    const dangerous = vi.fn().mockResolvedValue({ ...caution, scan_verdict: 'dangerous' })

    expect((await runPluginUpdate(dangerous, tern, null, p)).kind).toBe('scan')
    expect($confirmRequest.get()).toBeNull()
    expect(dangerous).toHaveBeenCalledTimes(1)
  })
})

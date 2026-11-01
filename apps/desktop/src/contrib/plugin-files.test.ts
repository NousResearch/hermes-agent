import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { notify } from '@/store/notifications'

import { createPluginContext } from './plugin'
import { revokeGrantsIfUninstalled } from './plugin-files'
import { $pluginRecords, type PluginRecord } from './plugins-store'

vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))

const door = {
  grants: vi.fn(async () => []),
  op: vi.fn(async () => ({ ok: true, text: 'hi' })),
  request: vi.fn(),
  revoke: vi.fn(async () => ({ ok: true, removed: 1 }))
}

beforeEach(() => {
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = { pluginFiles: door }
})

afterEach(() => {
  vi.clearAllMocks()
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

describe('ctx.files', () => {
  it("always names the calling plugin and asks with the app's own picker copy", async () => {
    door.request.mockResolvedValueOnce({ ok: true, folder: { grantedAt: 'x', id: 'f1', name: 'Vault', path: '/v' } })
    const files = createPluginContext('plexus').files

    expect(await files.requestFolder()).toEqual({ ok: true, folder: { id: 'f1', name: 'Vault', path: '/v' } })
    expect(door.request).toHaveBeenCalledWith('plexus', {
      buttonLabel: 'Allow access',
      title: 'Choose a folder for plexus'
    })

    await files.readText('f1', 'notes/a.md')
    expect(door.op).toHaveBeenCalledWith('plexus', { op: 'readText', folder: 'f1', path: 'notes/a.md' })
  })

  it('tells the user when the app refuses the folder they picked', async () => {
    door.request.mockResolvedValueOnce({ ok: false, error: 'protected', message: 'no' })

    expect(await createPluginContext('plexus').files.requestFolder()).toMatchObject({ ok: false, error: 'protected' })
    expect(notify).toHaveBeenCalledWith(expect.objectContaining({ kind: 'warning' }))
  })

  it('answers unavailable on a shell without the door instead of throwing', async () => {
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
    const files = createPluginContext('plexus').files

    expect(await files.requestFolder()).toMatchObject({ ok: false, error: 'unavailable' })
    expect(await files.list('data')).toMatchObject({ ok: false, error: 'unavailable' })
    expect(await files.folders()).toEqual([])
  })
})

describe('grants after a package removal', () => {
  afterEach(() => $pluginRecords.set({}))

  it('keeps the grants of a desktop plugin another profile still installs', async () => {
    $pluginRecords.set({ plexus: { id: 'plexus' } as PluginRecord })

    expect(await revokeGrantsIfUninstalled('plexus')).toBe(false)
    expect(door.revoke).not.toHaveBeenCalled()
  })

  it('drops the grants once the desktop plugin is gone', async () => {
    expect(await revokeGrantsIfUninstalled('plexus')).toBe(true)
    expect(door.revoke).toHaveBeenCalledWith('plexus')
  })

  it('a revoke that fails is reported, not counted as done', async () => {
    door.revoke.mockRejectedValueOnce(new Error('EACCES: grants.json'))

    await expect(revokeGrantsIfUninstalled('plexus')).rejects.toThrow('EACCES')
  })
})

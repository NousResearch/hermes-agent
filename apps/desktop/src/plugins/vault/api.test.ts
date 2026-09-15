import { beforeEach, describe, expect, it, vi } from 'vitest'

import { bindApi, configureVaultRoot, fetchVaultStatus } from './api'

describe('Vault plugin API', () => {
  const rest = vi.fn()

  beforeEach(() => {
    rest.mockReset()
    bindApi({ rest })
  })

  it('reads status through the plugin RPC boundary', async () => {
    rest.mockResolvedValue({ status: 'ok', root: 'C:\\Knowledge', notes_count: 3 })
    await expect(fetchVaultStatus()).resolves.toEqual({
      status: 'ok',
      root: 'C:\\Knowledge',
      notes_count: 3
    })
    expect(rest).toHaveBeenCalledWith('/status', undefined)
  })

  it('persists custom roots through the backend owner', async () => {
    rest.mockResolvedValue({ status: 'ok', root: 'D:\\Obsidian', notes_count: 42 })
    await configureVaultRoot('D:\\Obsidian')
    expect(rest).toHaveBeenCalledWith('/root', {
      method: 'PUT',
      body: { root: 'D:\\Obsidian' }
    })
  })
})

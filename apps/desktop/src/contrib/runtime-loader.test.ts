import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  getStatus: vi.fn()
}))

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal()),
  getStatus: mocks.getStatus
}))

import { $connection } from '@/store/session'

import { discoverRuntimePlugins } from './runtime-loader'

describe('runtime disk-plugin discovery', () => {
  const readDir = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = { readDir }
    $connection.set(null)
  })

  afterEach(() => {
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
    $connection.set(null)
  })

  it('does not probe a remote gateway home through the local Electron filesystem', async () => {
    mocks.getStatus.mockResolvedValue({ hermes_home: '/remote/hermes' })
    $connection.set({ mode: 'remote' } as never)

    await discoverRuntimePlugins()

    expect(mocks.getStatus).not.toHaveBeenCalled()
    expect(readDir).not.toHaveBeenCalled()
  })

  it('keeps scanning the local Hermes home for local gateways', async () => {
    mocks.getStatus.mockResolvedValue({ hermes_home: '/local/hermes' })
    readDir.mockResolvedValue({ entries: [] })
    $connection.set({ mode: 'local' } as never)

    await discoverRuntimePlugins()

    expect(readDir).toHaveBeenCalledWith('/local/hermes/desktop-plugins')
  })
})

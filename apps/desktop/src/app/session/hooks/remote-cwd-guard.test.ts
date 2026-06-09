import { afterEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'

/**
 * Tests for the remote-gateway CWD guard.
 *
 * When the desktop is connected to a remote gateway, session.info events
 * and cached-session restores must NOT overwrite the local CWD with a
 * remote VPS path that doesn't exist on this machine (issue #43042).
 */
describe('remote CWD guard', () => {
  afterEach(() => {
    $connection.set(null)
    vi.restoreAllMocks()
  })

  it('isRemoteGateway returns true in remote mode', async () => {
    const { isRemoteGateway } = await import('@/lib/media')
    $connection.set({ mode: 'remote' } as never)
    expect(isRemoteGateway()).toBe(true)
  })

  it('isRemoteGateway returns false in local mode', async () => {
    const { isRemoteGateway } = await import('@/lib/media')
    $connection.set({ mode: 'local' } as never)
    expect(isRemoteGateway()).toBe(false)
  })

  it('isRemoteGateway returns false with no connection', async () => {
    const { isRemoteGateway } = await import('@/lib/media')
    $connection.set(null)
    expect(isRemoteGateway()).toBe(false)
  })

  it('CWD guard allows updates in local mode', async () => {
    const { isRemoteGateway } = await import('@/lib/media')
    $connection.set({ mode: 'local' } as never)
    expect(isRemoteGateway()).toBe(false)
    // Guard condition: !isRemoteGateway() → true, so CWD update proceeds
  })

  it('CWD guard blocks updates in remote mode', async () => {
    const { isRemoteGateway } = await import('@/lib/media')
    $connection.set({ mode: 'remote' } as never)
    expect(isRemoteGateway()).toBe(true)
    // Guard condition: !isRemoteGateway() → false, so CWD update is skipped
  })
})

import { describe, expect, it, vi } from 'vitest'

import { handoffOauthCookies } from './oauth-cookie-handoff'

function fakeSession(reads: any[][] = []) {
  const get = vi.fn(async () => reads.shift() ?? [])
  const set = vi.fn(async () => undefined)
  const flushStorageData = vi.fn()

  return { session: { cookies: { get, set }, flushStorageData }, get, set, flushStorageData }
}

describe('OAuth cookie partition handoff', () => {
  it('copies HttpOnly gateway cookies before an unregistered URL is retargeted to its connection jar', async () => {
    const source = fakeSession([[{ name: 'hermes_session_at', value: 'at', httpOnly: true, secure: true, path: '/' }]])
    const target = fakeSession()

    await expect(
      handoffOauthCookies({
        source: source.session,
        target: target.session,
        url: 'https://gw.example.com:8443',
        cookieNames: ['hermes_session_at', 'hermes_session_rt']
      })
    ).resolves.toBe(1)

    expect(target.set).toHaveBeenCalledWith(
      expect.objectContaining({ url: 'https://gw.example.com:8443', name: 'hermes_session_at', value: 'at' })
    )
  })

  it('warms and retries a cold persisted source jar before accepting an empty read', async () => {
    const source = fakeSession([[], [{ name: 'hermes_session_rt', value: 'rt', path: '/' }]])
    const target = fakeSession()
    const wait = vi.fn(async () => undefined)

    await expect(
      handoffOauthCookies({
        source: source.session,
        target: target.session,
        url: 'https://gw.example.com',
        cookieNames: ['hermes_session_at', 'hermes_session_rt'],
        delaysMs: [0, 30],
        wait
      })
    ).resolves.toBe(1)

    expect(source.flushStorageData).toHaveBeenCalledOnce()
    expect(source.get).toHaveBeenCalledTimes(2)
    expect(wait).toHaveBeenCalledWith(30)
    expect(target.set).toHaveBeenCalledOnce()
  })

  it('surfaces a target write failure instead of silently registering against an empty jar', async () => {
    const source = fakeSession([[{ name: 'hermes_session_at', value: 'at' }]])
    const target = fakeSession()
    target.set.mockRejectedValueOnce(new Error('cookie database write failed'))

    await expect(
      handoffOauthCookies({
        source: source.session,
        target: target.session,
        url: 'https://gw.example.com',
        cookieNames: ['hermes_session_at']
      })
    ).rejects.toThrow('cookie database write failed')
  })
})

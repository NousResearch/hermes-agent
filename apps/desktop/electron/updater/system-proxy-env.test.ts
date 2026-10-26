import { describe, expect, it } from 'vitest'

import { LOCAL_BYPASS, restoreUserProxyEnv, systemProxyEnv } from './system-proxy-env'

const pac = (answers: Record<string, string>) => async (url: string) => answers[new URL(url).hostname] ?? 'DIRECT'
const proxied = (value: string) => ({ HTTPS_PROXY: value, https_proxy: value, HTTP_PROXY: value, http_proxy: value })

describe('update child proxy env from the system proxy (#123387)', () => {
  it('hands the system proxy to children that read only proxy env, bypassing hosts the system sends direct', async () => {
    const env = await systemProxyEnv(
      { PATH: '/usr/bin' },
      pac({ 'github.com': 'PROXY 10.0.0.7:7890; DIRECT', 'pypi.org': 'PROXY 10.0.0.7:7890' }),
      ['github.com', 'registry.npmjs.org', 'pypi.org']
    )

    const noProxy = [...LOCAL_BYPASS, 'registry.npmjs.org'].join(',')

    expect(env).toEqual({
      ...proxied('http://10.0.0.7:7890'),
      NO_PROXY: noProxy,
      no_proxy: noProxy,
      HERMES_UPDATE_SYSTEM_PROXY: 'HTTPS_PROXY,https_proxy,HTTP_PROXY,http_proxy,NO_PROXY,no_proxy'
    })
  })

  it('bypasses loopback, private ranges and .local names a PAC sends direct', async () => {
    const env = await systemProxyEnv({}, pac({ 'github.com': 'PROXY 10.0.0.7:7890' }), ['github.com'])

    for (const entry of ['127.0.0.0/8', '10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16', '::1', '.local']) {
      expect(env.NO_PROXY?.split(',')).toContain(entry)
    }
  })

  it('leaves an explicit proxy env alone and adds nothing when the system goes direct', async () => {
    const asked: string[] = []

    const resolve = async (url: string) => {
      asked.push(url)

      return 'PROXY 10.0.0.9:1080'
    }

    expect(await systemProxyEnv({ https_proxy: 'http://corp:3128' }, resolve)).toEqual({})
    expect(asked).toEqual([])
    expect(await systemProxyEnv({}, async () => 'DIRECT')).toEqual({})
  })

  it('exports no SOCKS or TLS proxy, which urllib and npm cannot use, and says so', async () => {
    for (const answer of ['SOCKS 10.0.0.9:1080', 'SOCKS5 10.0.0.9:1080', 'HTTPS 10.0.0.9:443']) {
      const notes: string[] = []

      expect(await systemProxyEnv({}, async () => answer, ['github.com'], note => notes.push(note))).toEqual({})
      expect(notes.join()).toContain(`github.com=${answer.split(' ')[0]}`)
    }
  })

  it('never routes one host through the proxy the PAC chose for another', async () => {
    const notes: string[] = []

    const env = await systemProxyEnv(
      {},
      pac({ 'github.com': 'PROXY 10.0.0.7:7890', 'pypi.org': 'PROXY 10.0.0.8:3128' }),
      ['github.com', 'pypi.org'],
      note => notes.push(note)
    )

    expect(env).toEqual({})
    expect(notes).toHaveLength(1)
  })

  it("extends the user's own NO_PROXY instead of replacing it, and restores it for the relaunched app", async () => {
    const user = { PATH: '/usr/bin', NO_PROXY: 'corp.internal' }
    const env = await systemProxyEnv(user, pac({ 'github.com': 'PROXY 10.0.0.7:7890' }), ['github.com'])

    expect(env.NO_PROXY).toBe(['corp.internal', ...LOCAL_BYPASS].join(','))
    expect(env.no_proxy).toBe(env.NO_PROXY)
    expect(restoreUserProxyEnv({ ...user, ...env })).toEqual(user)
  })

  it('never reads a proxy an earlier update marked as the user setting', async () => {
    const first = await systemProxyEnv({}, pac({ 'github.com': 'PROXY 10.0.0.7:7890' }), ['github.com'])
    // The system proxy moved; an env still carrying the first answer must not pin it.
    const next = await systemProxyEnv(first, pac({ 'github.com': 'PROXY 10.0.0.42:8080' }), ['github.com'])

    expect(next.HTTPS_PROXY).toBe('http://10.0.0.42:8080')
  })
})

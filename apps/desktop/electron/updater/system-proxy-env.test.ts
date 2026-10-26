import { describe, expect, it } from 'vitest'

import { systemProxyEnv } from './system-proxy-env'

const pac = (answers: Record<string, string>) => async (url: string) => answers[new URL(url).hostname] ?? 'DIRECT'

describe('update child proxy env from the system proxy (#123387)', () => {
  it('hands the system proxy to children that read only proxy env, bypassing hosts the system sends direct', async () => {
    const env = await systemProxyEnv(
      { PATH: '/usr/bin' },
      pac({ 'github.com': 'PROXY 10.0.0.7:7890; DIRECT', 'pypi.org': 'PROXY 10.0.0.7:7890' }),
      ['github.com', 'registry.npmjs.org', 'pypi.org']
    )

    expect(env).toEqual({
      HTTPS_PROXY: 'http://10.0.0.7:7890',
      https_proxy: 'http://10.0.0.7:7890',
      HTTP_PROXY: 'http://10.0.0.7:7890',
      http_proxy: 'http://10.0.0.7:7890',
      NO_PROXY: 'localhost,127.0.0.1,::1,registry.npmjs.org',
      no_proxy: 'localhost,127.0.0.1,::1,registry.npmjs.org'
    })
  })

  it('leaves an explicit proxy env alone and adds nothing when the system goes direct', async () => {
    const asked: string[] = []

    const resolve = async (url: string) => {
      asked.push(url)

      return 'SOCKS5 10.0.0.9:1080'
    }

    expect(await systemProxyEnv({ https_proxy: 'http://corp:3128' }, resolve)).toEqual({})
    expect(asked).toEqual([])
    expect(await systemProxyEnv({}, async () => 'DIRECT')).toEqual({})
    expect((await systemProxyEnv({}, resolve, ['github.com'])).HTTPS_PROXY).toBe('socks5h://10.0.0.9:1080')
  })
})

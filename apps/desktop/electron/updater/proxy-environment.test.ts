import { afterEach, expect, it } from 'vitest'

import { sourceUpdateProxyEnvironment } from './proxy-environment'

const proxyKeys = ['HTTPS_PROXY', 'https_proxy', 'HTTP_PROXY', 'http_proxy', 'ALL_PROXY', 'all_proxy'] as const
const saved = Object.fromEntries(proxyKeys.map(key => [key, process.env[key]]))

afterEach(() => {
  for (const key of proxyKeys) {
    const value = saved[key]

    if (value === undefined) {
      delete process.env[key]
    } else {
      process.env[key] = value
    }
  }
})

it('bridges Electron system proxy for a GUI updater with no proxy env', () => {
  expect(sourceUpdateProxyEnvironment('PROXY 127.0.0.1:7890; DIRECT', {})).toEqual({
    HTTPS_PROXY: 'http://127.0.0.1:7890'
  })
})

it('preserves explicit operator proxy environment', () => {
  expect(
    sourceUpdateProxyEnvironment('PROXY system.example:8080', {
      HTTPS_PROXY: 'http://operator.example:3128'
    })
  ).toEqual({})
})

it.each(['ELECTRON_MIRROR', 'ELECTRON_NIGHTLY_MIRROR'] as const)(
  'does not route an explicit %s through the system proxy',
  key => {
    expect(
      sourceUpdateProxyEnvironment('PROXY system.example:8080', {
        [key]: 'https://mirror.example/electron/'
      })
    ).toEqual({})
  }
)

it.each(['DIRECT', 'SOCKS5 127.0.0.1:1080', ''])('does not rewrite unsupported proxy result %j', resolved => {
  expect(sourceUpdateProxyEnvironment(resolved, {})).toEqual({})
})

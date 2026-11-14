import { expect, it } from 'vitest'

import { hasProxyEnv, isGitNetworkError, withoutProxyEnv } from './update-proxy'

it('disables inherited proxy spellings without dropping unrelated environment', () => {
  const env = withoutProxyEnv({ Https_Proxy: 'http://proxy', PATH: '/bin', NO_PROXY: 'localhost' })

  expect(hasProxyEnv(env)).toBe(false)
  expect(env.Https_Proxy).toBe('')
  expect(env.HTTPS_PROXY).toBe('')
  expect(env.PATH).toBe('/bin')
  expect(env.NO_PROXY).toBe('localhost')
})

it('detects mixed-case inherited proxies', () => {
  expect(hasProxyEnv({ Https_Proxy: 'http://proxy' })).toBe(true)
})

it('distinguishes network transport errors from missing branches', () => {
  expect(isGitNetworkError('fatal: unable to access https://github.com: Failed to connect')).toBe(true)
  expect(isGitNetworkError('fatal: Remote branch missing not found')).toBe(false)
})

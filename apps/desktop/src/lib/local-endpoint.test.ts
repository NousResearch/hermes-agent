import { describe, expect, it } from 'vitest'

import { isLoopbackEndpoint } from './local-endpoint'

describe('isLoopbackEndpoint', () => {
  it('recognizes the loopback hosts a local server listens on', () => {
    expect(isLoopbackEndpoint('http://localhost:11434/v1')).toBe(true)
    expect(isLoopbackEndpoint('http://127.0.0.1:8080/v1')).toBe(true)
    expect(isLoopbackEndpoint('https://127.9.8.7:8443/v1?x=1')).toBe(true)
    expect(isLoopbackEndpoint('http://[::1]:8080')).toBe(true)
    expect(isLoopbackEndpoint('::1')).toBe(true)
    expect(isLoopbackEndpoint('0.0.0.0:8000')).toBe(true)
    expect(isLoopbackEndpoint('my-box.localhost:1234')).toBe(true)
    expect(isLoopbackEndpoint('  http://LOCALHOST:1234/v1  ')).toBe(true)
  })

  it('leaves a provider on another machine alone', () => {
    expect(isLoopbackEndpoint('https://api.openai.com/v1')).toBe(false)
    expect(isLoopbackEndpoint('https://api.deepseek.com')).toBe(false)
    expect(isLoopbackEndpoint('http://192.168.1.5:8080/v1')).toBe(false)
    expect(isLoopbackEndpoint('http://127.example.com/v1')).toBe(false)
    expect(isLoopbackEndpoint('http://homelab.localhost.example/v1')).toBe(false)
  })

  it('answers false for nothing usable instead of throwing', () => {
    expect(isLoopbackEndpoint(null)).toBe(false)
    expect(isLoopbackEndpoint(undefined)).toBe(false)
    expect(isLoopbackEndpoint('')).toBe(false)
    expect(isLoopbackEndpoint('   ')).toBe(false)
    expect(isLoopbackEndpoint(8080)).toBe(false)
    expect(isLoopbackEndpoint('http://')).toBe(false)
    expect(isLoopbackEndpoint('not a url')).toBe(false)
  })
})

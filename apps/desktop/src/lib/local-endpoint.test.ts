import { describe, expect, it } from 'vitest'
import { isLocalEndpointUrl } from './local-endpoint'

describe('isLocalEndpointUrl', () => {
  it('classifies loopback and localhost variants as local', () => {
    expect(isLocalEndpointUrl('http://localhost:11434/v1')).toBe(true)
    expect(isLocalEndpointUrl('http://127.0.0.1:8080')).toBe(true)
    expect(isLocalEndpointUrl('http://127.8.9.10:11434')).toBe(true)
    expect(isLocalEndpointUrl('http://[::1]:11434/v1')).toBe(true)
    expect(isLocalEndpointUrl('http://0.0.0.0:11434')).toBe(true)
    expect(isLocalEndpointUrl('https://LOCALHOST:11434/v1')).toBe(true)
  })

  it('classifies RFC-1918 and link-local ranges as local', () => {
    expect(isLocalEndpointUrl('http://10.0.0.5:11434/v1')).toBe(true)
    expect(isLocalEndpointUrl('http://172.16.1.20:11434')).toBe(true)
    expect(isLocalEndpointUrl('http://172.31.255.1')).toBe(true)
    expect(isLocalEndpointUrl('http://192.168.1.10:11434/v1')).toBe(true)
    expect(isLocalEndpointUrl('http://169.254.169.254/latest/meta-data')).toBe(true)
  })

  it('classifies Tailscale CGNAT as local', () => {
    expect(isLocalEndpointUrl('http://100.100.100.100:11434')).toBe(true)
    expect(isLocalEndpointUrl('http://100.127.0.1')).toBe(true)
  })

  it('classifies mDNS, container-internal and unqualified hosts as local', () => {
    expect(isLocalEndpointUrl('http://byron.local:11434/v1')).toBe(true)
    expect(isLocalEndpointUrl('http://byron:11434/v1')).toBe(true)
    expect(isLocalEndpointUrl('http://ollama.docker.internal:11434')).toBe(true)
  })

  it('classifies IPv6 ULA and link-local as local', () => {
    expect(isLocalEndpointUrl('http://[fd00::1]:11434')).toBe(true)
    expect(isLocalEndpointUrl('http://[fe80::1]:11434')).toBe(true)
  })

  it('rejects public endpoints, empties and garbage', () => {
    expect(isLocalEndpointUrl('https://api.openai.com/v1')).toBe(false)
    expect(isLocalEndpointUrl('https://api.anthropic.com')).toBe(false)
    expect(isLocalEndpointUrl('https://api.openrouter.ai/api/v1')).toBe(false)
    expect(isLocalEndpointUrl('http://8.8.8.8/v1')).toBe(false)
    expect(isLocalEndpointUrl('http://[2607:f8b0::1]:11434')).toBe(false)
    expect(isLocalEndpointUrl('')).toBe(false)
    expect(isLocalEndpointUrl('   ')).toBe(false)
    expect(isLocalEndpointUrl('not a url')).toBe(false)
  })
})

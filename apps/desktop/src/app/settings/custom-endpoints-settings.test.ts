import { describe, expect, it, vi } from 'vitest'

// The component module reaches its data helpers through '@/hermes'; the pure
// payload mapper under test needs none of them, so stub the module to keep the
// import light and avoid touching the network layer.
vi.mock('@/hermes', () => ({
  activateCustomEndpoint: vi.fn(),
  deleteCustomEndpoint: vi.fn(),
  getCustomEndpoints: vi.fn(),
  saveCustomEndpoint: vi.fn(),
  validateCustomEndpoint: vi.fn()
}))

import { toPayload } from './custom-endpoints-settings'

function form(overrides: Record<string, unknown> = {}) {
  return {
    apiKey: '',
    baseUrl: ' http://192.168.1.69:11434/v1 ',
    contextLength: '',
    discoverModels: true,
    id: 'nasty',
    makeDefault: true,
    model: 'qwen3:8b',
    name: 'NASty',
    ...overrides
  } as Parameters<typeof toPayload>[0]
}

describe('toPayload', () => {
  it('sends an explicit 0 for a blank Context field so the backend clears the pin', () => {
    // Omitting the key would read as "unchanged" on the backend and leave a stale
    // override on disk, so the panel would show the old value again on reload.
    expect(toPayload(form({ contextLength: '' })).context_length).toBe(0)
  })

  it('treats a literal 0 or unparseable input as auto', () => {
    expect(toPayload(form({ contextLength: '0' })).context_length).toBe(0)
    expect(toPayload(form({ contextLength: 'abc' })).context_length).toBe(0)
  })

  it('passes a positive override through', () => {
    expect(toPayload(form({ contextLength: '65536' })).context_length).toBe(65536)
  })
})

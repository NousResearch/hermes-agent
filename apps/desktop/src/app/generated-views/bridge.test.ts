import { beforeEach, describe, expect, it, vi } from 'vitest'

import { createGeneratedViewBridge, GENERATED_VIEW_STATE_MAX_BYTES } from './bridge'

function bridge(overrides: Partial<Parameters<typeof createGeneratedViewBridge>[0]> = {}) {
  return createGeneratedViewBridge({
    capabilities: ['theme:read', 'state:persist'],
    bindings: ['hermes:status'],
    connectionKey: 'local:',
    id: 'demo',
    resolveBinding: vi.fn(async () => ({ running: true })),
    resolveTheme: vi.fn(() => ({ '--theme-primary': '#00ff66' })),
    ...overrides
  })
}

describe('generated-view bridge', () => {
  beforeEach(() => window.localStorage.clear())

  it('drops malformed and unknown messages silently', () => {
    const instance = bridge()
    const post = vi.fn()

    instance.handleMessage({ v: 2, type: 'hermes:getTheme', requestId: '1' }, post)
    instance.handleMessage({ v: 1, type: 'hermes:network', requestId: '2' }, post)

    expect(post).not.toHaveBeenCalled()
  })

  it('gates theme reads before touching the resolver', () => {
    const resolveTheme = vi.fn(() => ({ secret: 'no' }))
    const instance = bridge({ capabilities: [], resolveTheme })
    const post = vi.fn()

    instance.handleMessage({ v: 1, type: 'hermes:getTheme', requestId: 'theme-1' }, post)

    expect(resolveTheme).not.toHaveBeenCalled()
    expect(post).toHaveBeenCalledWith(expect.objectContaining({ code: 'capability_denied', type: 'hermes:error' }))
  })

  it('returns only declared allowlisted bindings', async () => {
    const resolveBinding = vi.fn(async () => ({ running: true }))
    const instance = bridge({ resolveBinding })
    const post = vi.fn()

    instance.handleMessage(
      { v: 1, type: 'hermes:getData', requestId: 'data-denied', bindingId: 'hermes:usage-30d' },
      post
    )
    expect(resolveBinding).not.toHaveBeenCalled()
    expect(post).toHaveBeenCalledWith(expect.objectContaining({ code: 'binding_denied' }))

    instance.handleMessage({ v: 1, type: 'hermes:getData', requestId: 'data-ok', bindingId: 'hermes:status' }, post)
    await vi.waitFor(() => expect(resolveBinding).toHaveBeenCalledWith('hermes:status'))
    expect(post).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'hermes:data', requestId: 'data-ok', data: { running: true } })
    )
  })

  it('persists bounded state under the host-bound view identity', () => {
    const post = vi.fn()
    const instance = bridge()

    instance.handleMessage({ v: 1, type: 'hermes:setState', requestId: 'set-1', state: { selected: 7 } }, post)
    instance.handleMessage({ v: 1, type: 'hermes:getState', requestId: 'get-1' }, post)

    expect(post).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'hermes:state', requestId: 'get-1', state: { selected: 7 }, version: 1 })
    )

    instance.handleMessage(
      { v: 1, type: 'hermes:setState', requestId: 'too-big', state: 'x'.repeat(GENERATED_VIEW_STATE_MAX_BYTES) },
      post
    )
    expect(post).toHaveBeenCalledWith(expect.objectContaining({ code: 'payload_too_large', requestId: 'too-big' }))
  })

  it('does not let one view address another view state', () => {
    const postA = vi.fn()
    const postB = vi.fn()
    const a = bridge({ id: 'a' })
    const b = bridge({ id: 'b' })

    a.handleMessage({ v: 1, type: 'hermes:setState', requestId: 'a-set', state: { owner: 'a' }, id: 'b' }, postA)
    b.handleMessage({ v: 1, type: 'hermes:getState', requestId: 'b-get', id: 'a' }, postB)

    expect(postB).toHaveBeenCalledWith(expect.objectContaining({ state: null, version: 0 }))
  })
})

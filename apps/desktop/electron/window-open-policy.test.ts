// @vitest-environment node
import { describe, expect, it, vi } from 'vitest'

import { createWindowOpenHandler, describeDeniedUrl } from './window-open-policy'

describe('createWindowOpenHandler', () => {
  it('denies every request with no side effect (CVE-2026-70608 posture)', () => {
    const handler = createWindowOpenHandler()

    expect(handler({ url: 'https://anything.example' })).toEqual({ action: 'deny' })
    expect(handler({ url: 'file:///C:/x.html' })).toEqual({ action: 'deny' })
    expect(handler({ url: 'javascript:alert(1)' })).toEqual({ action: 'deny' })
  })

  it('reports the sanitized origin to the logging observer', () => {
    const onDenied = vi.fn()

    createWindowOpenHandler(onDenied)({ url: 'https://example.com/path?token=secret' })

    expect(onDenied).toHaveBeenCalledExactlyOnceWith('https://example.com')
  })

  it('a throwing logging observer cannot change the decision', () => {
    const handler = createWindowOpenHandler(() => {
      throw new Error('log failed')
    })

    expect(handler({ url: 'https://example.com/x' })).toEqual({ action: 'deny' })
  })
})

describe('describeDeniedUrl', () => {
  it('logs origin only, never the full URL', () => {
    expect(describeDeniedUrl('https://example.com/path?token=secret')).toBe('https://example.com')
    expect(describeDeniedUrl('data:text/html,foo')).toBe('data:')
    expect(describeDeniedUrl('not a url')).toBe('<unparseable>')
  })
})

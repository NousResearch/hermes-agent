import { describe, expect, it } from 'vitest'

import { formatErrorDiagnostics, parseErrorSurface } from './error-surface'

describe('parseErrorSurface', () => {
  it('accepts a valid descriptor', () => {
    expect(parseErrorSurface({ layer: 'streaming', code: 'stream_drop', retryable: true })).toEqual({
      layer: 'streaming',
      code: 'stream_drop',
      retryable: true
    })
  })

  it('accepts every documented layer', () => {
    for (const layer of ['provider', 'endpoint', 'streaming', 'auth', 'billing', 'gateway', 'runtime', 'disk']) {
      expect(parseErrorSurface({ layer, code: 'x', retryable: false })?.layer).toBe(layer)
    }
  })

  it('rejects unknown layers and non-objects', () => {
    expect(parseErrorSurface({ layer: 'blockchain', code: 'x', retryable: true })).toBeNull()
    expect(parseErrorSurface('provider')).toBeNull()
    expect(parseErrorSurface(null)).toBeNull()
    expect(parseErrorSurface(undefined)).toBeNull()
    expect(parseErrorSurface(7)).toBeNull()
  })

  it('defaults code and retryable when missing', () => {
    expect(parseErrorSurface({ layer: 'gateway' })).toEqual({ layer: 'gateway', code: 'unknown', retryable: true })
  })

  it('honors retryable=false', () => {
    expect(parseErrorSurface({ layer: 'auth', code: 'auth_permanent', retryable: false })?.retryable).toBe(false)
  })
})

describe('formatErrorDiagnostics', () => {
  it('includes layer, code, model and error', () => {
    const text = formatErrorDiagnostics({
      errorText: 'boom',
      model: 'anthropic/claude-opus-4.6',
      surface: { layer: 'provider', code: 'rate_limit', retryable: true }
    })

    expect(text).toContain('layer: provider')
    expect(text).toContain('code: rate_limit')
    expect(text).toContain('model: anthropic/claude-opus-4.6')
    expect(text).toContain('error: boom')
  })

  it('omits absent fields without leaving blank lines', () => {
    const text = formatErrorDiagnostics({ errorText: 'boom' })

    expect(text).not.toContain('layer:')
    expect(text).not.toContain('model:')
    expect(text.split('\n').every(line => line.trim().length > 0)).toBe(true)
  })
})

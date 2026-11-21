import { describe, expect, it } from 'vitest'

import { externalUrlTarget } from './external-url'

describe('externalUrlTarget', () => {
  it('opens web and mail URLs', () => {
    expect(externalUrlTarget('https://example.com/x')).toEqual({ kind: 'web', url: 'https://example.com/x' })
    expect(externalUrlTarget('mailto:a@example.com')?.kind).toBe('web')
  })

  it('opens a well-formed tel: number', () => {
    expect(externalUrlTarget('tel:+46104956404')).toEqual({ kind: 'web', url: 'tel:+46104956404' })
    expect(externalUrlTarget('tel:0104956404')).toEqual({ kind: 'web', url: 'tel:0104956404' })
  })

  it('refuses a tel: URL that is not a plain number', () => {
    for (const raw of ['tel:123&calc', 'tel:+46 10 495', 'tel:12345', 'tel:+4610495640412345', 'tel:abc1234567']) {
      expect(externalUrlTarget(raw)).toBeNull()
    }
  })

  it('refuses unsupported schemes', () => {
    expect(externalUrlTarget('javascript:alert(1)')).toBeNull()
    expect(externalUrlTarget('sms:+46104956404')).toBeNull()
  })
})

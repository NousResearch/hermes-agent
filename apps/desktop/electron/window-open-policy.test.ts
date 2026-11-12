// @vitest-environment node
import { describe, expect, it } from 'vitest'

import { createWindowOpenHandler, describeDeniedUrl } from './window-open-policy'

describe('createWindowOpenHandler', () => {
  it('denies every request (CVE-2026-70608 posture)', () => {
    const handler = createWindowOpenHandler()

    expect(handler({ url: 'https://anything.example' })).toEqual({ action: 'deny' })
  })

  it('a throwing logging observer cannot change the decision', () => {
    const handler = createWindowOpenHandler(() => {
      throw new Error('log failed')
    })

    expect(handler({ url: 'https://github.com/NousResearch/hermes-agent' })).toEqual({ action: 'deny' })
  })
})

describe('describeDeniedUrl', () => {
  it('logs origin only, never the full URL', () => {
    expect(describeDeniedUrl('https://example.com/path?token=secret')).toBe('https://example.com')
    expect(describeDeniedUrl('data:text/html,foo')).toBe('data:')
    expect(describeDeniedUrl('not a url')).toBe('<unparseable>')
  })
})

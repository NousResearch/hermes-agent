// A plain-HTTP page (the Webapp opened from another machine) has `crypto.getRandomValues` but no
// `crypto.randomUUID`: session.create still gets a request_id there.
import { afterEach, expect, it, vi } from 'vitest'

import { canonicalRequest } from './gateway-canonical'

afterEach(() => {
  vi.unstubAllGlobals()
})

it('mints the session.create request_id outside a secure context', () => {
  vi.stubGlobal('crypto', { getRandomValues: (bytes: Uint8Array) => bytes.fill(7) })

  const request = canonicalRequest('session.create', {}, { sources: ['tui'], parameters: ['source', 'request_id'] })

  expect(request.params).toEqual({ source: 'tui', request_id: '07070707-0707-4707-8707-070707070707' })
})

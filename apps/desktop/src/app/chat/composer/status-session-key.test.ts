import { describe, expect, it } from 'vitest'

import { statusStackSessionKey } from './status-session-key'

// The status stack polls `process.list` under whatever key it is handed. A
// stored messaging conversation (a Telegram chat opened in Desktop) can sit with
// NO live runtime id, and with `null` the stack never armed discovery at all —
// the background job started on the gateway side stayed invisible.
describe('statusStackSessionKey', () => {
  it('prefers the live runtime id whenever there is one', () => {
    expect(statusStackSessionKey('rt-1', 'stored-1')).toBe('rt-1')
  })

  it('falls back to the durable conversation key when no runtime is bound', () => {
    expect(statusStackSessionKey(null, 'stored-1')).toBe('stored-1')
  })

  it('trims the durable key so padding never becomes a scope of its own', () => {
    expect(statusStackSessionKey(null, '  stored-1  ')).toBe('stored-1')
  })

  it.each([null, undefined, '', '   '])('yields null when neither identity is usable (%p)', durable => {
    expect(statusStackSessionKey(null, durable)).toBeNull()
  })

  it('never lets a blank runtime id shadow a usable durable key', () => {
    expect(statusStackSessionKey('   ', 'stored-1')).toBe('stored-1')
  })
})

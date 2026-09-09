import { describe, expect, it } from 'vitest'
import { quickEntrySubmitAck } from './use-quick-entry-bridge'

describe('quickEntrySubmitAck', () => {
  it('reports a rejected prompt as failure instead of acknowledging success', () => {
    expect(quickEntrySubmitAck(false)).toEqual({
      code: 'submit-rejected',
      message: 'The prompt was not accepted.',
      ok: false,
      retryable: true
    })
  })

  it('acknowledges only an accepted prompt as success', () => {
    expect(quickEntrySubmitAck(true)).toEqual({ ok: true })
  })
})

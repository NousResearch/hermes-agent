import { describe, expect, it } from 'vitest'

import { isCodeSkewRestartRequired } from './code-skew-error'

const SKEW_IPC =
  'Error invoking remote method \'hermes:api\': Error: 503: {"detail":"Restart required: This process is running code from 08b4875f4a but the checkout on disk is now 48d2528066. The model picker would risk a stale-module crash — restart the Desktop-owned backend to load the new code (use Restart backend in Hermes Desktop, or quit and reopen the app)"}'

describe('isCodeSkewRestartRequired', () => {
  it('detects the Models-page IPC 503 from a stale Desktop-owned backend', () => {
    expect(isCodeSkewRestartRequired(new Error(SKEW_IPC))).toBe(true)
  })

  it('detects a bare FastAPI detail string', () => {
    expect(isCodeSkewRestartRequired('Restart required: checkout drifted')).toBe(true)
  })

  it('ignores unrelated 503s and load failures', () => {
    expect(isCodeSkewRestartRequired(new Error('503: Service Unavailable'))).toBe(false)
    expect(isCodeSkewRestartRequired(new Error('Failed to load models'))).toBe(false)
    expect(isCodeSkewRestartRequired(null)).toBe(false)
  })
})

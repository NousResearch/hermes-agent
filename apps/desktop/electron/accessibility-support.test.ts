import { describe, expect, it } from 'vitest'

import { shouldForceAccessibilitySupport } from './accessibility-support'

describe('shouldForceAccessibilitySupport', () => {
  it('forces accessibility support on Windows, where UI Automation dictation tools need it (#92607)', () => {
    expect(shouldForceAccessibilitySupport('win32')).toBe(true)
  })

  it('leaves it off on macOS and Linux, where the reported gap does not apply', () => {
    expect(shouldForceAccessibilitySupport('darwin')).toBe(false)
    expect(shouldForceAccessibilitySupport('linux')).toBe(false)
  })
})

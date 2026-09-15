import { describe, expect, it } from 'vitest'

import { isPackagedDesktopRuntime } from './runtime-mode'

describe('isPackagedDesktopRuntime', () => {
  it('honors the explicit E2E development override over Electron packaged mode', () => {
    expect(isPackagedDesktopRuntime(true, { HERMES_DESKTOP_FORCE_DEV: '1' })).toBe(false)
    expect(
      isPackagedDesktopRuntime(true, {
        HERMES_DESKTOP_FORCE_DEV: '1',
        HERMES_DESKTOP_IS_PACKAGED: '1',
      }),
    ).toBe(false)
  })

  it('uses Electron packaged mode when no E2E override is present', () => {
    expect(isPackagedDesktopRuntime(true, {})).toBe(true)
    expect(isPackagedDesktopRuntime(false, {})).toBe(false)
  })

  it('keeps the existing explicit packaged override for non-Electron runtimes', () => {
    expect(isPackagedDesktopRuntime(false, { HERMES_DESKTOP_IS_PACKAGED: '1' })).toBe(true)
  })
})

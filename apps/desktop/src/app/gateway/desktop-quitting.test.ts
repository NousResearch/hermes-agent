import { describe, expect, it } from 'vitest'

import { isIntentionalDesktopQuitError } from './desktop-quitting'

describe('isIntentionalDesktopQuitError', () => {
  it('matches the sealed-quit abort, including the Electron IPC wrapper', () => {
    expect(isIntentionalDesktopQuitError(new Error('Hermes Desktop is quitting.'))).toBe(true)
    expect(
      isIntentionalDesktopQuitError(
        new Error("Error invoking remote method 'hermes:connection': Error: Hermes Desktop is quitting.")
      )
    ).toBe(true)
  })

  it('does not swallow a real boot failure', () => {
    expect(isIntentionalDesktopQuitError(new Error('Timed out connecting to Hermes backend'))).toBe(false)
    expect(isIntentionalDesktopQuitError('backend exited')).toBe(false)
    expect(isIntentionalDesktopQuitError({ code: 1, signal: 'SIGTERM' })).toBe(false)
  })

  it('matches payload marked as intentionalTeardown or containing quit error message', () => {
    expect(isIntentionalDesktopQuitError({ intentionalTeardown: true })).toBe(true)
    expect(isIntentionalDesktopQuitError({ code: null, signal: null, error: 'Hermes Desktop is quitting.' })).toBe(true)
  })
})

import { describe, expect, it } from 'vitest'

import { stripIpcErrorPrefix } from './ipc-error'

describe('stripIpcErrorPrefix', () => {
  it('drops the Electron wrapper for custom Error subclasses', () => {
    expect(
      stripIpcErrorPrefix("Error invoking remote method 'hermes:api': DesktopBridgeError: The backend is unavailable.")
    ).toBe('The backend is unavailable.')
  })

  it('drops the Electron wrapper for plain Error instances', () => {
    expect(stripIpcErrorPrefix("Error invoking remote method 'hermes:api': Error: The request failed.")).toBe(
      'The request failed.'
    )
  })

  it('leaves unwrapped messages unchanged', () => {
    expect(stripIpcErrorPrefix('Desktop boot failed.')).toBe('Desktop boot failed.')
  })
})

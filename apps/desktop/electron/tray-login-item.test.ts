import { describe, expect, it, vi } from 'vitest'

import { applyLoginItemPreference, readLoginItemPreference } from './tray-login-item'

describe('tray login item coordination', () => {
  it('returns null when Windows login-item state cannot be read', () => {
    expect(
      readLoginItemPreference({
        getLoginItemSettings: () => {
          throw new Error('registry unavailable')
        }
      })
    ).toBeNull()
  })

  it('reports failure when Windows rejects a login-item change', () => {
    expect(
      applyLoginItemPreference(true, {
        setLoginItemSettings: () => {
          throw new Error('access denied')
        }
      })
    ).toBe(false)
  })

  it('writes the requested executable and reports success', () => {
    const setLoginItemSettings = vi.fn()

    expect(applyLoginItemPreference(true, { setLoginItemSettings }, 'C:\\Hermes\\Hermes.exe')).toBe(true)
    expect(setLoginItemSettings).toHaveBeenCalledWith({
      args: [],
      openAtLogin: true,
      path: 'C:\\Hermes\\Hermes.exe'
    })
  })
})

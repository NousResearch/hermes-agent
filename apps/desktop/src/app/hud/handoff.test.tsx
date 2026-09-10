import { cleanup, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const openSession = vi.fn()

vi.mock('../open-session', () => ({
  openSession: (...args: unknown[]) => openSession(...args)
}))

import { useHudGoto } from './handoff'

describe('useHudGoto', () => {
  afterEach(() => {
    cleanup()
    openSession.mockReset()
    Reflect.deleteProperty(window, 'hermesDesktop')
  })

  it('switches conversations through the canonical open-session door without forcing route navigation', () => {
    let onGoto: ((storedSessionId: string) => void) | undefined
    const navigate = vi.fn()

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        hud: {
          onGoto: vi.fn((callback: (storedSessionId: string) => void) => {
            onGoto = callback

            return vi.fn()
          })
        }
      }
    })

    renderHook(() => useHudGoto(navigate))
    onGoto?.('stored-session-b')

    expect(openSession).toHaveBeenCalledWith('stored-session-b', navigate)
    expect(navigate).not.toHaveBeenCalled()
  })
})

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $sessionDeliveryModes,
  sessionDeliveryModeFor,
  setSessionDeliveryMode,
  toggleSessionDeliveryMode
} from './session-delivery-mode'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop
const setBridge = vi.fn()
const mem = new Map<string, string>()

beforeEach(() => {
  mem.clear()
  $sessionDeliveryModes.set({})
  vi.stubGlobal('localStorage', {
    getItem: (k: string) => (mem.has(k) ? mem.get(k)! : null),
    setItem: (k: string, v: string) => {
      mem.set(k, String(v))
    },
    removeItem: (k: string) => {
      mem.delete(k)
    },
    clear: () => mem.clear(),
    key: () => null,
    length: 0
  })
  desktopWindow.hermesDesktop = {
    setSessionDeliveryMode: setBridge
  } as unknown as Window['hermesDesktop']
  setBridge.mockClear()
})

afterEach(() => {
  desktopWindow.hermesDesktop = initialHermesDesktop
  vi.unstubAllGlobals()
})

describe('session-delivery-mode store', () => {
  it('defaults off and scopes by sessionId', () => {
    expect(sessionDeliveryModeFor('a')).toBe('off')
    setSessionDeliveryMode('a', 'deep_premium')
    expect(sessionDeliveryModeFor('a')).toBe('deep_premium')
    expect(sessionDeliveryModeFor('b')).toBe('off')
    expect(setBridge).toHaveBeenLastCalledWith({ sessionId: 'a', mode: 'deep_premium' })
  })

  it('toggles and stays session-local', () => {
    expect(toggleSessionDeliveryMode('s1')).toBe('deep_premium')
    expect(toggleSessionDeliveryMode('s1')).toBe('off')
    setSessionDeliveryMode('s2', 'deep_premium')
    expect(sessionDeliveryModeFor('s1')).toBe('off')
    expect(sessionDeliveryModeFor('s2')).toBe('deep_premium')
  })
})

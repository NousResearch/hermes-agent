import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $autoSendIdleDelayMs,
  $autoSendIdleEnabled,
  setAutoSendIdleDelayId,
  setAutoSendIdleEnabled
} from './auto-send'

const ENABLED_KEY = 'hermes.desktop.autoSendIdle'
const DELAY_KEY = 'hermes.desktop.autoSendIdleDelay'

describe('auto-send preferences', () => {
  beforeEach(() => {
    localStorage.clear()
    $autoSendIdleEnabled.set(false)
    $autoSendIdleDelayMs.set(2000)
  })

  it('defaults to disabled and 2000ms delay when localStorage is empty', () => {
    expect($autoSendIdleEnabled.get()).toBe(false)
    expect($autoSendIdleDelayMs.get()).toBe(2000)
  })

  it('setAutoSendIdleEnabled(true) sets the atom and persists to localStorage', () => {
    setAutoSendIdleEnabled(true)
    expect($autoSendIdleEnabled.get()).toBe(true)
    expect(localStorage.getItem(ENABLED_KEY)).toBe('true')
  })

  it('setAutoSendIdleDelayId("3000") sets $autoSendIdleDelayMs to 3000 and persists the id string', () => {
    setAutoSendIdleDelayId('3000')
    expect($autoSendIdleDelayMs.get()).toBe(3000)
    expect(localStorage.getItem(DELAY_KEY)).toBe('3000')
  })

  it('falls back to 2000 when stored delay id is unknown or corrupt', async () => {
    localStorage.setItem(DELAY_KEY, 'nonsense')
    vi.resetModules()
    const fresh = await import('./auto-send')
    expect(fresh.$autoSendIdleDelayMs.get()).toBe(2000)
  })

  it('falls back to false when stored enabled value is non-boolean', async () => {
    localStorage.setItem(ENABLED_KEY, 'yes')
    vi.resetModules()
    const fresh = await import('./auto-send')
    expect(fresh.$autoSendIdleEnabled.get()).toBe(false)
  })
})

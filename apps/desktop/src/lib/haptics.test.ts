import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setHapticsMuted } from '@/store/haptics'

import {
  getHapticTelemetrySnapshot,
  registerHapticTrigger,
  resetHapticTelemetryForTests,
  triggerHaptic
} from './haptics'

describe('haptic telemetry', () => {
  beforeEach(() => {
    window.localStorage.clear()
    setHapticsMuted(false)
    registerHapticTrigger(null)
    resetHapticTelemetryForTests()
  })

  afterEach(() => {
    window.localStorage.clear()
    setHapticsMuted(false)
    registerHapticTrigger(null)
    resetHapticTelemetryForTests()
  })

  it('records haptic trigger outcomes only when haptics debug is enabled', () => {
    const trigger = vi.fn(() => Promise.resolve())
    registerHapticTrigger(trigger)

    triggerHaptic('streamStart')
    expect(getHapticTelemetrySnapshot()).toEqual({
      attempted: 0,
      failed: 0,
      fired: 0,
      missingTrigger: 0,
      muted: 0,
      rateLimited: 0
    })

    window.localStorage.setItem('hermes.desktop.hapticsDebug', 'true')
    triggerHaptic('streamStart')

    expect(trigger).toHaveBeenCalledTimes(2)
    expect(getHapticTelemetrySnapshot()).toEqual({
      attempted: 1,
      failed: 0,
      fired: 1,
      missingTrigger: 0,
      muted: 0,
      rateLimited: 0
    })
  })

  it('records muted and missing-trigger skips when debug telemetry is enabled', () => {
    window.localStorage.setItem('hermes.desktop.hapticsDebug', 'true')

    setHapticsMuted(true)
    triggerHaptic('warning')
    setHapticsMuted(false)

    triggerHaptic('warning')

    expect(getHapticTelemetrySnapshot()).toEqual({
      attempted: 2,
      failed: 0,
      fired: 0,
      missingTrigger: 1,
      muted: 1,
      rateLimited: 0
    })
  })
})

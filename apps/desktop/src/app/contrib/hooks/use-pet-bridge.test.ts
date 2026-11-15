import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import {
  $petActivity,
  $petSpriteTarget,
  $petState,
  petInteract,
  setPetActivity,
  setPetHandling,
  setPetInfo
} from '@/store/pet'
import { clearAllSessionStates, publishSessionState } from '@/store/session-states'

import { startPetNap, syncPetBusyFromSessions } from './use-pet-bridge'

const busySession = (storedId: string) => ({ ...createClientSessionState(storedId), busy: true })

describe('syncPetBusyFromSessions', () => {
  beforeEach(() => {
    clearAllSessionStates()
    setPetActivity({ busy: false, toolRunning: false, reasoning: false, awaitingInput: false })
  })

  afterEach(() => {
    clearAllSessionStates()
    setPetActivity({ busy: false, toolRunning: false, reasoning: false, awaitingInput: false })
  })

  it('marks the pet busy while any session is working, foreground or not', () => {
    // A background session is running — its per-session stream hook is NOT
    // mounted, so nothing ever set toolRunning/reasoning on $petActivity.
    publishSessionState('runtime-background', busySession('stored-background'))

    syncPetBusyFromSessions()

    expect($petActivity.get().busy).toBe(true)
    // busy maps to the `run` pose even without a foreground toolRunning signal.
    expect($petState.get()).toBe('run')
  })

  it('clears busy and returns to idle once every session is at rest', () => {
    publishSessionState('runtime-background', busySession('stored-background'))
    syncPetBusyFromSessions()
    expect($petActivity.get().busy).toBe(true)

    // Turn ends: the session is no longer busy.
    publishSessionState('runtime-background', {
      ...createClientSessionState('stored-background'),
      busy: false
    })

    syncPetBusyFromSessions()

    expect($petActivity.get().busy).toBe(false)
    expect($petState.get()).toBe('idle')
  })

  it('stays busy when only one of several sessions finishes', () => {
    publishSessionState('runtime-a', busySession('stored-a'))
    publishSessionState('runtime-b', busySession('stored-b'))
    syncPetBusyFromSessions()
    expect($petActivity.get().busy).toBe(true)

    // Session A finishes; B is still running.
    publishSessionState('runtime-a', { ...createClientSessionState('stored-a'), busy: false })
    syncPetBusyFromSessions()

    expect($petActivity.get().busy).toBe(true)
    expect($petState.get()).toBe('run')
  })
})

describe('startPetNap', () => {
  let stop = () => {}

  beforeEach(() => {
    vi.useFakeTimers()
    setPetActivity({ busy: false, toolRunning: false, reasoning: false, awaitingInput: false })
    setPetInfo({ enabled: true, behaviors: { sleepAfterSeconds: 10 } })
    stop = startPetNap()
  })

  afterEach(() => {
    stop()
    setPetHandling(null)
    setPetActivity({ busy: false, sleepy: false })
    setPetInfo({ enabled: false })
    vi.useRealTimers()
  })

  it('resizing a sleeping pet does not wake it; a new sleep delay re-arms', () => {
    vi.advanceTimersByTime(10_000)
    setPetInfo({ enabled: true, scale: 0.5, behaviors: { sleepAfterSeconds: 10 } })
    expect($petState.get()).toBe('sleep')

    setPetInfo({ enabled: true, scale: 0.5, behaviors: { sleepAfterSeconds: 20 } })
    expect($petState.get()).toBe('idle')
    vi.advanceTimersByTime(20_000)
    expect($petState.get()).toBe('sleep')
  })

  // Each pass re-sleeps after the delay, so this also proves a touch re-arms the nap timer.
  it('any interaction wakes it, even on a pet with no handling rows to play', () => {
    for (const kind of ['poke', 'pickup', 'drop', 'land'] as const) {
      vi.advanceTimersByTime(10_000)
      expect($petState.get()).toBe('sleep')

      petInteract(kind)
      expect($petSpriteTarget.get().state).toBe('idle')
      expect($petActivity.get().sleepy).toBe(false)
    }
  })
})

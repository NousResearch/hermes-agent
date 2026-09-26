import { afterEach, expect, test, vi } from 'vitest'

import { $desktopOrbConnection } from '@/app/jarvis/desktop-orb-state'
import { publishMicLevel } from '@/store/voice-level'

import { initPetOverlayBridge, type PetOverlayControl, popInPet, popOutDesktopOrb } from './pet-overlay'

afterEach(() => {
  popInPet()
  vi.useRealTimers()
})

test('orb keeps its shared control channel after the in-window pet unmounts and mirrors measured audio', async () => {
  vi.useFakeTimers()
  let receive: ((event: PetOverlayControl) => void) | undefined
  const disconnect = vi.fn()
  const pushState = vi.fn()
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { petOverlay: {
    onControl: (handler: (event: PetOverlayControl) => void) => { receive = handler;

 return disconnect },
    open: vi.fn(async () => ({ ok: true })), close: vi.fn(async () => ({ ok: true })), pushState
  } } })
  const releasePet = initPetOverlayBridge()
  const releaseOrb = initPetOverlayBridge()
  $desktopOrbConnection.set({ active: true, connected: true, locale: 'pl' })
  popOutDesktopOrb()
  await Promise.resolve()
  releasePet()
  expect(disconnect).not.toHaveBeenCalled()
  receive?.({ type: 'ready' })
  expect(pushState.mock.lastCall?.[0].orb.active).toBe(true)
  publishMicLevel(0.5)
  await vi.advanceTimersByTimeAsync(50)
  expect(pushState.mock.lastCall?.[0].audioLevel).toBe(0.5)
  releaseOrb()
  expect(disconnect).toHaveBeenCalledTimes(1)
})

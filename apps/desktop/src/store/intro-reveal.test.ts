import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $introReveal,
  finishIntroReveal,
  hasSeenIntroReveal,
  installIntroRevealBridgeListeners,
  shouldPlayFirstRunIntro,
  startIntroReveal
} from './intro-reveal'

beforeEach(() => {
  window.localStorage.clear()
  $introReveal.set({ phase: 'hidden' })
})

afterEach(() => vi.unstubAllGlobals())

describe('intro reveal', () => {
  it('requires the launch flag, an unseen intro and no explicit first-run skip', () => {
    const open = vi.fn().mockResolvedValue({ ok: true })
    const close = vi.fn().mockResolvedValue({ ok: true })

    for (const seen of [false, true]) {
      if (seen) {
        $introReveal.set({ phase: 'playing' })
        finishIntroReveal()
      }

      for (const flag of [undefined, false, true]) {
        vi.stubGlobal('hermesDesktop', { guestOnboardingEnabled: flag, introReveal: { open, close } })

        for (const skipped of [false, true]) {
          expect(shouldPlayFirstRunIntro(skipped)).toBe(flag === true && !seen && !skipped)
        }

        if (flag !== true) {
          startIntroReveal()
          expect(open).not.toHaveBeenCalled()
          expect($introReveal.get().phase).toBe('hidden')
        }
      }
    }
  })

  it('finishes synchronously and restores the app after a native skip or close', () => {
    let skipped: (() => void) | undefined
    let closed: (() => void) | undefined
    const open = vi.fn().mockResolvedValue({ ok: true })

    const close = vi.fn(() => {
      expect(hasSeenIntroReveal()).toBe(true)
      expect($introReveal.get().phase).toBe('hidden')

      return Promise.resolve({ ok: true })
    })

    vi.stubGlobal('hermesDesktop', {
      guestOnboardingEnabled: true,
      introReveal: {
        open,
        close,
        onSkip: (callback: () => void) => {
          skipped = callback

          return () => {
            skipped = undefined
          }
        },
        onClosed: (callback: () => void) => {
          closed = callback

          return () => {
            closed = undefined
          }
        }
      }
    })

    const dispose = installIntroRevealBridgeListeners()

    startIntroReveal()
    expect(open).toHaveBeenCalledWith({ hideMain: true })
    skipped?.()
    expect($introReveal.get().phase).toBe('leaving')
    closed?.()
    expect(close).toHaveBeenCalledWith({ showMain: true })
    expect(finishIntroReveal()).toBeUndefined()
    expect(close).toHaveBeenCalledTimes(1)
    dispose()
  })
})

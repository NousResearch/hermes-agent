import { atom } from 'nanostores'

import { isOnboardingEnabled } from '@/lib/onboarding-enabled'
import { readKey, writeKey } from '@/lib/storage'
import { setOnboardingSurfaceActive } from '@/store/onboarding-presence'

const SEEN_KEY = 'hermes-intro-reveal-seen-v1'

export type IntroRevealPhase = 'hidden' | 'playing' | 'leaving'

export interface IntroRevealState {
  phase: IntroRevealPhase
}

const INITIAL: IntroRevealState = { phase: 'hidden' }

export const $introReveal = atom<IntroRevealState>(INITIAL)

$introReveal.subscribe(state => setOnboardingSurfaceActive('intro', state.phase !== 'hidden'))

export function hasSeenIntroReveal(): boolean {
  return readKey(SEEN_KEY) === '1'
}

export function isIntroRevealEnabled(): boolean {
  return isOnboardingEnabled()
}

export function shouldPlayFirstRunIntro(firstRunSkipped: boolean): boolean {
  return isIntroRevealEnabled() && !firstRunSkipped && !hasSeenIntroReveal()
}

export function startIntroReveal(): void {
  if (!isIntroRevealEnabled() || $introReveal.get().phase !== 'hidden') {
    return
  }

  $introReveal.set({ phase: 'playing' })
  // The film plays over the desktop; every exit path must restore the app.
  void window.hermesDesktop?.introReveal?.open({ hideMain: true }).catch(finishIntroReveal)
}

export function leaveIntroReveal(): void {
  if ($introReveal.get().phase === 'playing') {
    $introReveal.set({ phase: 'leaving' })
  }
}

export function finishIntroReveal(): void {
  if ($introReveal.get().phase === 'hidden') {
    return
  }

  writeKey(SEEN_KEY, '1')
  $introReveal.set(INITIAL)
  void window.hermesDesktop?.introReveal?.close({ showMain: true }).catch(() => undefined)
}

export function installIntroRevealBridgeListeners(): () => void {
  const bridge = window.hermesDesktop?.introReveal
  const offSkip = bridge?.onSkip(leaveIntroReveal)
  const offClosed = bridge?.onClosed(finishIntroReveal)

  return () => {
    offSkip?.()
    offClosed?.()
  }
}

import '@nous-research/ui/styles/fonts.css'

import { createRoot } from 'react-dom/client'

import { OverlayErrorBoundary } from '@/components/overlay-error-boundary'
import { isOnboardingEnabled } from '@/lib/onboarding-enabled'

import { IntroRevealSurface } from './intro-reveal-surface'

export function mountIntroReveal(): void {
  if (!isOnboardingEnabled()) {
    return
  }

  document.title = 'Hermes'
  // Every intro measure is in rem, so this one root size scales the whole
  // composition. The app's default 16 px root is sized for a working window, which
  // is too small on a display the user sits back from.
  document.documentElement.style.fontSize = '150%'
  const root = document.getElementById('root')

  if (!root) {
    return
  }

  // StrictMode would double-start this disposable window's clock and sound.
  createRoot(root).render(
    <OverlayErrorBoundary label="intro-reveal">
      <IntroRevealSurface />
    </OverlayErrorBoundary>
  )

  // Native ready-to-show can precede the first React paint.
  requestAnimationFrame(() =>
    requestAnimationFrame(() => {
      window.hermesDesktop?.introReveal?.ready()
    })
  )
}

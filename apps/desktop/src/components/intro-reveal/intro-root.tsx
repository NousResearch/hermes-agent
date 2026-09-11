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

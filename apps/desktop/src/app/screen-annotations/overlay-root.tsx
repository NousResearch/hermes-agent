import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import { ErrorBoundary } from '@/components/error-boundary'

import { ScreenAnnotationsApp } from './annotations-app'

/**
 * Boot the screen-annotation overlay window. Loaded by the same bundle as the
 * main app but via `?win=annotate`, so it shares build plumbing while mounting
 * a minimal, transparent surface (no app shell, no gateway, no theme — the
 * marks use fixed screen-legible colors).
 *
 * index.html skips the opaque anti-flash backing for this window kind (a
 * full-display sheet painted #f7f7f7 is a white veil over the user's screen).
 * styles.css still assigns `body` a themed background, so we also force every
 * host layer transparent here — same trick as the pet overlay.
 */
export function mountScreenAnnotations(): void {
  // Drop the inline boot color too — a stylesheet `background` rule does not
  // always beat `element.style.backgroundColor` on the first composite, and
  // this window is the size of the display.
  document.documentElement.style.backgroundColor = 'transparent'
  if (document.body) {
    document.body.style.backgroundColor = 'transparent'
  }

  const style = document.createElement('style')
  style.textContent =
    'html,body,#root{background:transparent !important;background-color:transparent !important;}'
  document.head.appendChild(style)

  const root = document.getElementById('root')

  if (!root) {
    return
  }

  createRoot(root).render(
    <StrictMode>
      <ErrorBoundary label="screen-annotations">
        <ScreenAnnotationsApp />
      </ErrorBoundary>
    </StrictMode>
  )
}

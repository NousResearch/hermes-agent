import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import { ErrorBoundary } from '@/components/error-boundary'
import { ThemeProvider } from '@/themes/context'

import { PetOverlayApp } from './pet-overlay-app'

/**
 * Boot the pet-overlay window. Loaded by the same bundle as the main app but
 * via `?win=overlay`, so it shares CSS/atoms while mounting a minimal, transparent
 * surface (no app shell, no gateway, no I18n — the bubble strings are inline).
 *
 * The index.html boot script paints an OPAQUE themed background to avoid a flash
 * in normal windows; the overlay must be see-through, so we force every host
 * layer transparent with a late, high-specificity style tag.
 */
export function mountPetOverlay(): void {
  const style = document.createElement('style')
  style.textContent = 'html,body,#root{background:transparent !important;}'
  document.head.appendChild(style)

  const root = document.getElementById('root')

  if (!root) {
    return
  }

  createRoot(root).render(
    <StrictMode>
      <ErrorBoundary label="pet-overlay">
        <ThemeProvider>
          <PetOverlayApp />
        </ThemeProvider>
      </ErrorBoundary>
    </StrictMode>
  )
}

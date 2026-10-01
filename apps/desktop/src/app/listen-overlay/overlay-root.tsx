import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import { ErrorBoundary } from '@/components/error-boundary'
import { ThemeProvider } from '@/themes/context'

import { ListenOverlayApp } from './listen-overlay-app'

/**
 * Boot the listen-overlay window (`?win=listen`). Same split as the pet
 * overlay: a gateway-less transparent surface that mirrors state pushed from
 * main over IPC — see electron/listen-overlay.ts.
 */
export function mountListenOverlay(): void {
  const style = document.createElement('style')
  style.textContent = 'html,body,#root{background:transparent !important;}'
  document.head.appendChild(style)

  const root = document.getElementById('root')

  if (!root) {
    return
  }

  createRoot(root).render(
    <StrictMode>
      <ErrorBoundary label="listen-overlay">
        <ThemeProvider>
          <ListenOverlayApp />
        </ThemeProvider>
      </ErrorBoundary>
    </StrictMode>
  )
}

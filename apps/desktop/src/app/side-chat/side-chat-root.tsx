import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import { ErrorBoundary } from '@/components/error-boundary'
import { ThemeProvider } from '@/themes/context'

import { SideChatApp } from './side-chat-app'

/**
 * Boot the side chat window. Loaded by the same bundle as the main app but via
 * `?win=side`, so it shares CSS/theme tokens while mounting a minimal aside
 * surface (no app shell, no gateway, no router).
 *
 * The index.html boot script paints an OPAQUE themed background to avoid a
 * flash in normal windows; this window is a floating card on a transparent
 * backdrop, so force the host layers see-through (same trick as Quick Entry).
 */
export function mountSideChat(): void {
  const style = document.createElement('style')
  style.textContent = 'html,body,#root{background:transparent !important;}'
  document.head.appendChild(style)

  const root = document.getElementById('root')

  if (!root) {
    return
  }

  createRoot(root).render(
    <StrictMode>
      <ErrorBoundary label="side-chat">
        <ThemeProvider>
          <SideChatApp />
        </ThemeProvider>
      </ErrorBoundary>
    </StrictMode>
  )
}

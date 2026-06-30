import '@/styles.css'
// Side-effect: applies the persisted window translucency on load.
import '@/store/translucency'

import { QueryClientProvider } from '@tanstack/react-query'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { HashRouter } from 'react-router-dom'

import App from '@/app'
import { ErrorBoundary } from '@/components/error-boundary'
import { HapticsProvider } from '@/components/haptics-provider'
import { I18nProvider } from '@/i18n'
import { installClipboardShim } from '@/lib/clipboard'
import { queryClient } from '@/lib/query-client'
import { setPaneOpen } from '@/store/panes'
import { ThemeProvider } from '@/themes/context'

// ponytail: mark mobile-standalone so desktop's mobile gates engage.
if (typeof window !== 'undefined') {
  ;(window as unknown as { __HERMES_MOBILE_STANDALONE__?: boolean }).__HERMES_MOBILE_STANDALONE__ = true
  document.documentElement.classList.add('hermes-mobile-standalone')
  // Persistent overlay/route state from previous sessions can land the WebView
  // deep inside the app. Reset to root so the boot view is always the intro.
  if (window.location.hash && window.location.hash !== '#/') window.location.hash = ''
}

// Mobile: collapse desktop's docked sidebar + right-rail at boot so the empty
// intro owns the screen. Reopen via the hamburger / pane controls on demand.
setPaneOpen('chat-sidebar', false)
setPaneOpen('preview', false)

// Reset master-detail drill state whenever the route changes (closing an
// overlay, navigating to root, opening a different overlay).
if (typeof window !== 'undefined') {
  window.addEventListener('hashchange', () => {
    document.body.removeAttribute('data-mobile-drilled')
  })
}








installClipboardShim()

if (import.meta.env.MODE !== 'production') {
  import('@/app/chat/perf-probe')
}

if (new URLSearchParams(window.location.search).get('win') === 'overlay') {
  void import('@/app/pet-overlay/overlay-root').then(({ mountPetOverlay }) => mountPetOverlay())
} else {
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <ErrorBoundary label="root">
        <QueryClientProvider client={queryClient}>
          <I18nProvider>
            <ThemeProvider>
              <HapticsProvider>
                <HashRouter>
                  <App />
                </HashRouter>
              </HapticsProvider>
            </ThemeProvider>
          </I18nProvider>
        </QueryClientProvider>
      </ErrorBoundary>
    </StrictMode>
  )
}


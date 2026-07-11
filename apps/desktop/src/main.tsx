import './styles.css'
// Side-effect: applies the persisted window translucency on load.
import './store/translucency'

import { QueryClientProvider } from '@tanstack/react-query'
import { StrictMode, useEffect } from 'react'
import { createRoot } from 'react-dom/client'
import { HashRouter } from 'react-router-dom'

import App from './app'
import { ErrorBoundary } from './components/error-boundary'
import { HapticsProvider } from './components/haptics-provider'
import { I18nProvider } from './i18n'
import { installClipboardShim } from './lib/clipboard'
import { queryClient } from './lib/query-client'
import { ThemeProvider } from './themes/context'

installClipboardShim()

function RendererReadySignal() {
  useEffect(() => {
    const frame = window.requestAnimationFrame(() => window.hermesDesktop?.signalRendererReady?.())
    return () => window.cancelAnimationFrame(frame)
  }, [])

  return null
}

// Dev-only: install __PERF_DRIVE__ + __PERF_PROBE__ on window so the
// scripts/ harnesses can drive a synthetic stream + record render cost.
// Tree-shaken out of production builds. (Uses MODE rather than DEV because
// our Vite setup currently bundles with PROD=true even in `vite dev`; see
// scripts/dev-no-hmr.mjs for the surrounding workarounds.)
if (import.meta.env.MODE !== 'production') {
  import('./app/chat/perf-probe')
}

// The pet overlay rides this same bundle (`?win=overlay`) but mounts a tiny,
// transparent, gateway-less surface instead of the full app. Branch before any
// app-shell work so the overlay window stays cheap.
if (new URLSearchParams(window.location.search).get('win') === 'overlay') {
  void import('./app/pet-overlay/overlay-root').then(({ mountPetOverlay }) => mountPetOverlay())
} else {
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <ErrorBoundary label="root">
        <QueryClientProvider client={queryClient}>
          <I18nProvider>
            <ThemeProvider>
              <HapticsProvider>
                <HashRouter>
                  <RendererReadySignal />
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

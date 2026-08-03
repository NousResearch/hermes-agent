import './styles.css'
// Side-effect: applies the persisted window translucency on load.
import './store/translucency'

import { QueryClientProvider } from '@tanstack/react-query'
import { StrictMode } from 'react'
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

// The perf probe ships in dev, and in a production build ONLY when explicitly
// opted in (VITE_PERF_PROBE=1) — this lets the perf harness measure a real,
// minified production renderer for representative absolute numbers. Normal
// `npm run build` leaves the flag unset, so the probe never reaches users.
if (import.meta.env.MODE !== 'production' || import.meta.env.VITE_PERF_PROBE === '1') {
  import('./app/chat/perf-probe')
}

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
                {/* useTransitions={false}: react-router v7's HashRouter wraps every
                    route state update in React.startTransition() by default. In
                    React 19's concurrent renderer, transitions are non-urgent — React
                    can yield mid-render and resume later. When the app is under load
                    (streaming token deltas, gateway events, store updates), those
                    higher-priority updates keep interrupting the transition, starving
                    the route change commit. The session sidebar highlight + main pane
                    both freeze for seconds despite the main thread being free.
                    Disabling transitions makes navigate() commit at default priority. */}
                <HashRouter useTransitions={false}>
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

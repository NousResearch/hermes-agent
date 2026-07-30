import './styles.css'
// Side-effect: reports in-flight turns to the main process for the quit guard.
import './store/active-work'
// Side-effect: applies the persisted window translucency on load.
import './store/translucency'
// Dev-only render/state churn counters. MUST precede the `react-dom` import
// below: react-dom captures the devtools hook at module init, so bippy has to
// install during THIS import's evaluation or every commit goes unseen
// (verified — a late install reports renderers=0, commits=0). `vite.config.ts`
// aliases this specifier to a no-op module for non-dev builds, so neither the
// counters nor bippy reach a shipped renderer.
import '@/debug/dev-only'

import { QueryClientProvider } from '@tanstack/react-query'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { HashRouter } from 'react-router'

import App from './app'
import { ErrorBoundary } from './components/error-boundary'
import { HapticsProvider } from './components/haptics-provider'
import { RootTooltipProvider } from './components/ui/tooltip'
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

const winParam = new URLSearchParams(window.location.search).get('win')

if (winParam === 'overlay') {
  void import('./app/pet-overlay/overlay-root').then(({ mountPetOverlay }) => mountPetOverlay())
} else if (winParam === 'quick') {
  void import('./app/quick-entry/quick-entry-root').then(({ mountQuickEntry }) => mountQuickEntry())
} else {
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <ErrorBoundary label="root">
        <QueryClientProvider client={queryClient}>
          <I18nProvider>
            <ThemeProvider>
              <HapticsProvider>
                {/* ONE tooltip provider for the whole app. Every `Tip` used to
                    carry its own, and with ~107 call sites those subtrees
                    dominated unrelated interactions (52,784 TooltipProvider
                    renders in a single sash drag). Radix's provider holds only
                    refs and stable callbacks, so hoisting is what it's for. */}
                <RootTooltipProvider>
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
                </RootTooltipProvider>
              </HapticsProvider>
            </ThemeProvider>
          </I18nProvider>
        </QueryClientProvider>
      </ErrorBoundary>
    </StrictMode>
  )
}

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

if (import.meta.env.MODE !== 'production') {
  import('./app/chat/perf-probe')
}

const windowKind = new URLSearchParams(window.location.search).get('win')

document.documentElement.dataset.windowKind = windowKind ?? 'app'
document.body.dataset.windowKind = windowKind ?? 'app'

if (windowKind === 'menu-bar-companion') {
  document.documentElement.style.backgroundColor = 'transparent'
  document.body.style.backgroundColor = 'transparent'
}

// Helper windows ride this same bundle but mount small surfaces instead of the
// full app. Branch before app-shell work so they stay cheap.
if (windowKind === 'overlay') {
  void import('./app/pet-overlay/overlay-root').then(({ mountPetOverlay }) => mountPetOverlay())
} else if (windowKind === 'menu-bar-companion') {
  void import('./app/menu-bar-companion/shell').then(({ MenuBarCompanionShell }) => {
    createRoot(document.getElementById('root')!).render(
      <StrictMode>
        <ErrorBoundary label="menu-bar-companion-root">
          <ThemeProvider>
            <MenuBarCompanionShell />
          </ThemeProvider>
        </ErrorBoundary>
      </StrictMode>
    )
  })
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

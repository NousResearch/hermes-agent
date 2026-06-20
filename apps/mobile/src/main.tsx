// 1) Install the window.hermesDesktop bridge BEFORE any vendored desktop module
//    evaluates (this import must stay first — see bridge/boot.ts).
import '~bridge/boot'

// 2) Design system + mobile structural CSS. ./styles.css inlines the desktop
//    design system AND adds the Tailwind @source for the desktop renderer.
import './styles.css'
import '~mobile/theme-fallback.css'

// 2b) Mobile-only wording overrides (touch gestures vs mouse/keyboard). Side
//     effect; must run before the I18nProvider reads the catalog.
import '~mobile/i18n-overrides'

// 3) Desktop side-effect: persisted translucency (native calls are optional-chained).
import '@/store/translucency'

import { QueryClientProvider } from '@tanstack/react-query'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { HashRouter } from 'react-router-dom'

// Reuse the desktop's own providers so the mobile app shares its exact runtime:
// query cache, i18n, theme tokens, haptics, routing.
import { ErrorBoundary } from '@/components/error-boundary'
import { HapticsProvider } from '@/components/haptics-provider'
import { I18nProvider } from '@/i18n'
import { installClipboardShim } from '@/lib/clipboard'
import { queryClient } from '@/lib/query-client'
import { ThemeProvider } from '@/themes/context'

import { MobileRoot } from './app'

installClipboardShim()

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary label="root">
      <QueryClientProvider client={queryClient}>
        <I18nProvider>
          <ThemeProvider>
            <HapticsProvider>
              <HashRouter>
                <MobileRoot />
              </HashRouter>
            </HapticsProvider>
          </ThemeProvider>
        </I18nProvider>
      </QueryClientProvider>
    </ErrorBoundary>
  </StrictMode>,
)

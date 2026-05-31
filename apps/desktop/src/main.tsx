import './styles.css'

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { HashRouter } from 'react-router-dom'

import App from './app'
import { HapticsProvider } from './components/haptics-provider'
import { installClipboardShim } from './lib/clipboard'
import { isDemoMode } from './lib/demo-flag'
import { ThemeProvider } from './themes/context'

installClipboardShim()

// Dev-only: install __PERF_DRIVE__ + __PERF_PROBE__ on window so the
// scripts/ harnesses can drive a synthetic stream + record render cost.
// Tree-shaken out of production builds. (Uses MODE rather than DEV because
// our Vite setup currently bundles with PROD=true even in `vite dev`; see
// scripts/dev-no-hmr.mjs for the surrounding workarounds.)
if (import.meta.env.MODE !== 'production') {
  import('./app/chat/perf-probe')
}

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      staleTime: 60_000
    }
  }
})

async function bootstrap() {
  // Opt-in demo / fixture mode: run against canned data + a fake gateway (no
  // backend). Dynamically imported so production builds tree-shake it out.
  if (isDemoMode()) {
    const { installDemo } = await import('./demo')
    installDemo()
  }

  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <QueryClientProvider client={queryClient}>
        <ThemeProvider>
          <HapticsProvider>
            <HashRouter>
              <App />
            </HashRouter>
          </HapticsProvider>
        </ThemeProvider>
      </QueryClientProvider>
    </StrictMode>
  )
}

void bootstrap()

import '../../src/styles.css'

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import { ManagedRolloutsSection } from '../../src/app/settings/managed-rollouts/managed-rollouts-section'
import { I18nProvider } from '../../src/i18n/context'
import { ThemeProvider } from '../../src/themes/context'

interface RehearsalWindow extends Window {
  __managedRolloutRows?: unknown[]
  __managedRolloutCalls?: { preparation: number; start: number; command: number }
}

const host = window as RehearsalWindow
const calls = { preparation: 0, start: 0, command: 0 }

host.__managedRolloutCalls = calls

Object.defineProperty(window, 'hermesDesktop', {
  configurable: true,
  value: {
    connections: {
      managedRollouts: {
        capabilities: async () => ({ protocol: 1, available: true, reason: null, maxConcurrency: 1, maxInstallations: 500 }),
        inventory: async () => ({
          inventoryRevision: 'ui-rehearsal-1',
          capturedMono: 100,
          observations: host.__managedRolloutRows ?? []
        }),
        activeRevision: async () => null,
        history: async () => ({ items: [], nextCursor: null }),
        resolveTarget: async () => { throw new Error('Target resolution is outside the UI rehearsal.') },
        preflight: async () => { throw new Error('Preflight is outside the UI rehearsal.') },
        start: async () => {
          calls.start += 1
          throw new Error('Start is forbidden in the UI rehearsal.')
        },
        read: async () => ({ revision: 0, snapshot: null }),
        get: async () => null,
        command: async () => {
          calls.command += 1
          throw new Error('Commands are forbidden in the UI rehearsal.')
        }
      },
      updateManaged: async () => {
        calls.preparation += 1
        throw new Error('Preparation is forbidden in the UI rehearsal.')
      }
    }
  }
})

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <I18nProvider configClient={null} initialLocale="ar">
      <ThemeProvider>
        <main className="mx-auto min-h-screen max-w-5xl p-6">
          <ManagedRolloutsSection />
        </main>
      </ThemeProvider>
    </I18nProvider>
  </StrictMode>
)

import '../../src/styles.css'

import { StrictMode, useState } from 'react'
import { createRoot } from 'react-dom/client'

import { OverlayView } from '../../src/app/overlays/overlay-view'
import { ManagedRolloutsSection } from '../../src/app/settings/managed-rollouts/managed-rollouts-section'
import { ConfirmDialog } from '../../src/components/ui/confirm-dialog'
import { I18nProvider } from '../../src/i18n/context'
import { ThemeProvider } from '../../src/themes/context'

interface RehearsalWindow extends Window {
  __managedRolloutRows?: unknown[]
  __managedRolloutCalls?: { preparation: number; start: number; command: number }
  __managedRolloutJourney?: { overlayClose: number; dialogClose: number; dialogConfirm: number }
}

const host = window as RehearsalWindow
const calls = { preparation: 0, start: 0, command: 0 }

host.__managedRolloutCalls = calls

// Focus-journey counters live for the page lifetime so a test can read them
// after each Escape, independent of any component re-render.
const journey = { overlayClose: 0, dialogClose: 0, dialogConfirm: 0 }

host.__managedRolloutJourney = journey

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

/**
 * Modal focus-return journey harness: mounts the production OverlayView (its
 * own Escape layer) with the production ConfirmDialog (a Radix dialog that
 * owns Escape while open). Each layer records its close so a test can prove
 * one Escape closes exactly one layer, focus returns to the trigger, and no
 * background navigation happens.
 */
function FocusJourney() {
  const [dialogOpen, setDialogOpen] = useState(false)
  const [overlayOpen, setOverlayOpen] = useState(true)

  if (!overlayOpen) {
    return <p id="overlay-closed">overlay-closed</p>
  }

  return (
    <OverlayView
      closeLabel="close-overlay"
      onClose={() => {
        journey.overlayClose += 1
        setOverlayOpen(false)
      }}
    >
      <div className="p-6">
        <button id="open-confirm" onClick={() => setDialogOpen(true)} type="button">
          open-confirm
        </button>
        <ConfirmDialog
          confirmLabel="confirm-journey"
          onClose={() => {
            journey.dialogClose += 1
            setDialogOpen(false)
          }}
          onConfirm={() => {
            journey.dialogConfirm += 1
          }}
          open={dialogOpen}
          title="focus-journey-dialog"
        />
      </div>
    </OverlayView>
  )
}

const focusJourney = new URLSearchParams(window.location.search).get('journey') === 'focus'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <I18nProvider configClient={null} initialLocale="ar">
      <ThemeProvider>
        {focusJourney ? (
          <FocusJourney />
        ) : (
          <main className="mx-auto min-h-screen max-w-5xl p-6">
            <ManagedRolloutsSection />
          </main>
        )}
      </ThemeProvider>
    </I18nProvider>
  </StrictMode>
)

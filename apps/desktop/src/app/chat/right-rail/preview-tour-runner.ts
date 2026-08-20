/**
 * PREVIEW TOUR RUNNER REGISTRY — the tour tool's script bridge into the
 * preview pane, the tour analog of preview-nav's handle registry.
 *
 * A live browser pane registers a script runner (its webview's
 * `executeJavaScript`) here, keyed by tab id; `activeTourRunner` resolves the
 * ACTIVE tab from the store. Kept separate from preview-tour.ts so the pane
 * component's static import stays tiny — the engine source + driver.js
 * payload only load when a tour actually runs (gateway-event dynamic-imports
 * preview-tour.ts).
 */

import { $rightRailActiveTabId } from '@/store/layout'
import { $previewTabs } from '@/store/preview'

/** Runs JS source in the pane's guest page, resolving its completion value. */
export type TourScriptRunner = (code: string) => Promise<unknown>

const runners = new Map<string, TourScriptRunner>()

/** Register a live preview's script runner; returns an idempotent unregister. */
export function registerPreviewTourRunner(tabId: string, runner: TourScriptRunner): () => void {
  runners.set(tabId, runner)

  return () => {
    if (runners.get(tabId) === runner) {
      runners.delete(tabId)
    }
  }
}

/** The ACTIVE preview tab's script runner. Null = no live page to tour. */
export function activeTourRunner(): TourScriptRunner | null {
  const tabs = $previewTabs.get()
  const tab = tabs.find(t => t.id === $rightRailActiveTabId.get()) ?? tabs[0]

  return (tab && runners.get(tab.id)) || null
}

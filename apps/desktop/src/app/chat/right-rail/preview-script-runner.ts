/**
 * PREVIEW SCRIPT RUNNER REGISTRY — the one way anything in the app reaches into
 * the preview pane's guest page, the script analog of preview-nav's handle
 * registry.
 *
 * A live browser pane registers its webview's `executeJavaScript` here, keyed
 * by tab id; `activePreviewScriptRunner` resolves the ACTIVE tab from the
 * store. Both guest-page features ride it — the tour tool (preview-tour.ts)
 * and the interaction tool (preview-act.ts) — so their heavy payloads stay out
 * of the pane component's static import graph and only load when used.
 */

import { $rightRailActiveTabId } from '@/store/layout'
import { $previewTabs } from '@/store/preview'

/** Runs JS source in the pane's guest page, resolving its completion value. */
export type PreviewScriptRunner = (code: string) => Promise<unknown>

const runners = new Map<string, PreviewScriptRunner>()

/** Register a live preview's script runner; returns an idempotent unregister. */
export function registerPreviewScriptRunner(tabId: string, runner: PreviewScriptRunner): () => void {
  runners.set(tabId, runner)

  return () => {
    if (runners.get(tabId) === runner) {
      runners.delete(tabId)
    }
  }
}

/** The ACTIVE preview tab's script runner. Null = no live page behind it. */
export function activePreviewScriptRunner(): PreviewScriptRunner | null {
  const tabs = $previewTabs.get()
  const tab = tabs.find(t => t.id === $rightRailActiveTabId.get()) ?? tabs[0]

  return (tab && runners.get(tab.id)) || null
}

/** Every mounted guest-page runner. Idle cleanup is a renderer-wide invariant:
 * a tab switch must not strand the pulse that another tab started. */
export function previewScriptRunners(): readonly PreviewScriptRunner[] {
  return [...runners.values()]
}

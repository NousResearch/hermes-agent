/**
 * PREVIEW SCRIPT RUNNER REGISTRY — the one way anything in the app reaches into
 * the preview pane's guest page, the script analog of preview-nav's handle
 * registry.
 *
 * A live browser pane registers its webview's `executeJavaScript` here, keyed
 * by tab id; `activePreviewScriptRunner` resolves the ACTIVE tab among the
 * requesting session's tabs. Both guest-page features ride it — the tour tool (preview-tour.ts)
 * and the interaction tool (preview-act.ts) — so their heavy payloads stay out
 * of the pane component's static import graph and only load when used.
 */

import type { PreviewOwner } from '@/store/preview-ownership'

import { activePreviewTabFor } from './preview-active-tab'

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

/** The script runner of the ACTIVE tab among those `owner` (the requesting
 *  session's stored id; omitted = the focused session) may see. Null = no live
 *  page behind it. */
export function activePreviewScriptRunner(owner?: PreviewOwner): PreviewScriptRunner | null {
  const tab = activePreviewTabFor(owner)

  return (tab && runners.get(tab.id)) || null
}

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

import { resolveActivePreviewTab } from './preview-reader'

/** Runs JS source in the pane's guest page, resolving its completion value. */
export type PreviewScriptRunner = (code: string) => Promise<unknown>

export interface ActivePreviewScriptRunner {
  generation: number
  runner: PreviewScriptRunner
  tabId: string
}

interface RegisteredRunner {
  generation: number
  ready: boolean
  runner: PreviewScriptRunner
}

const runners = new Map<string, RegisteredRunner>()
let nextDocumentGeneration = 0

/** Register a live preview's script runner; returns an idempotent unregister. */
export function registerPreviewScriptRunner(
  tabId: string,
  runner: PreviewScriptRunner,
  options: { initiallyReady?: boolean } = {}
): () => void {
  const registration = { generation: ++nextDocumentGeneration, ready: options.initiallyReady ?? true, runner }

  runners.set(tabId, registration)

  return () => {
    if (runners.get(tabId) === registration) {
      runners.delete(tabId)
    }
  }
}

/** The ACTIVE preview tab's script runner. Null = no live page behind it. */
export function activePreviewScriptRunner(): PreviewScriptRunner | null {
  const tabs = $previewTabs.get()
  const tab = tabs.find(t => t.id === $rightRailActiveTabId.get()) ?? tabs[0]

  return (tab && runners.get(tab.id)?.runner) || null
}

/** Retire bindings to the current guest document before its replacement loads. */
export function invalidatePreviewScriptRunner(tabId: string): void {
  const registration = runners.get(tabId)

  if (registration) {
    registration.ready = false
    registration.generation = ++nextDocumentGeneration
  }
}

/** Mark the live renderer as ready after Electron created its new document. */
export function markPreviewDocumentReady(tabId: string): void {
  const registration = runners.get(tabId)

  if (registration) {
    registration.ready = true
    registration.generation = ++nextDocumentGeneration
  }
}

/** Follow the Electron guest's main-frame replacement and ready boundaries. */
export function watchPreviewDocumentLifecycle(target: EventTarget, tabId: string): () => void {
  const onStartNavigation = (event: Event) => {
    const detail = event as Event & { isInPlace?: boolean; isMainFrame?: boolean }

    if (detail.isMainFrame === true && detail.isInPlace !== true) {
      invalidatePreviewScriptRunner(tabId)
    }
  }

  const onReady = () => markPreviewDocumentReady(tabId)

  target.addEventListener('did-start-navigation', onStartNavigation)
  target.addEventListener('dom-ready', onReady)

  return () => {
    target.removeEventListener('did-start-navigation', onStartNavigation)
    target.removeEventListener('dom-ready', onReady)
  }
}

/** The page the preview reader considers selected, paired with its exact live runner. */
export function resolveActivePreviewScriptRunner(): ActivePreviewScriptRunner | null {
  const tab = resolveActivePreviewTab()
  const registration = tab ? runners.get(tab.id) : undefined

  return tab && registration?.ready
    ? { generation: registration.generation, runner: registration.runner, tabId: tab.id }
    : null
}

/** True only while the selected tab still has the same registered runner. */
export function isActivePreviewScriptRunner(binding: ActivePreviewScriptRunner): boolean {
  const active = resolveActivePreviewScriptRunner()

  return active?.tabId === binding.tabId && active.runner === binding.runner && active.generation === binding.generation
}

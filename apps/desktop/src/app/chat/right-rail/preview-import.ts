/**
 * Import-from-the-web handles, one per live browser pane.
 *
 * pen.dev's capturer runs in main against the pane's <webview> guest, so all
 * the renderer contributes is the guest's identity and what the page is right
 * now. Registered by `PreviewPane`; read by the strip's import control and by
 * the agent's `pen_canvas(action='import')`, which works on the ACTIVE tab.
 */

import { $rightRailActiveTabId } from '@/store/layout'
import { $previewTabs } from '@/store/preview'

export interface PreviewImportHandle {
  /** The <webview>'s webContents id, once attached. */
  guestId: () => number | undefined
  /** True while the guest is between `did-start-loading` and `did-stop-loading`. */
  loading: () => boolean
  page: () => { title: string; url: string }
}

const handles = new Map<string, PreviewImportHandle>()

export function registerPreviewImport(tabId: string, handle: PreviewImportHandle): () => void {
  handles.set(tabId, handle)

  return () => {
    if (handles.get(tabId) === handle) {
      handles.delete(tabId)
    }
  }
}

export function previewImportHandle(tabId: string): PreviewImportHandle | null {
  return handles.get(tabId) ?? null
}

/** The ACTIVE preview tab's handle — the page the user is looking at. */
export function activePreviewImport(): { handle: PreviewImportHandle; tabId: string } | null {
  const tabs = $previewTabs.get()
  const tab = tabs.find(t => t.id === $rightRailActiveTabId.get()) ?? tabs[0]
  const handle = tab && handles.get(tab.id)

  return tab && handle ? { handle, tabId: tab.id } : null
}

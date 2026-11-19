import { atom } from 'nanostores'

import { $poppedBrowserTabIds, $previewTabs, openOwnedBrowserTab } from './preview'
import { currentSessionId } from './preview-ownership'
import { $focusedStoredSessionId } from './session-focus'
import { isBrowserWindow } from './windows'

/** Ephemeral intent, never persisted; the addressed pane consumes it once. */
export const $previewDesignRequest = atom<{ sessionId: null | string; tabId: string } | null>(null)

export type PreviewDesignOpenResult = 'another-conversation' | 'ok' | 'popped-out' | 'unavailable'

export function openFocusedBrowserInDesignMode(ownerSessionId: null | string): PreviewDesignOpenResult {
  if (isBrowserWindow()) {
    return 'popped-out'
  }

  const focusedSessionId = $focusedStoredSessionId.get()

  if (focusedSessionId !== ownerSessionId) {
    return 'another-conversation'
  }

  const activeId = openOwnedBrowserTab(focusedSessionId)
  const tab = $previewTabs.get().find(item => item.id === activeId)

  if (!tab || tab.target.kind !== 'url') {
    return 'unavailable'
  }

  if ($poppedBrowserTabIds.get().has(tab.id)) {
    return 'popped-out'
  }

  if (tab.sessionId && currentSessionId(tab.sessionId) !== currentSessionId(focusedSessionId)) {
    return 'another-conversation'
  }

  $previewDesignRequest.set({ sessionId: focusedSessionId, tabId: tab.id })

  return 'ok'
}

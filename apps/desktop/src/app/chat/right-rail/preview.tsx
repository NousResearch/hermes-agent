import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { $restartPreviewServer } from '@/app/contrib/panes'
import { $previewReloadRequest, $previewTabs, adoptPersistedBrowserTab, previewTabInAnyProfile } from '@/store/preview'
import { isBrowserWindow } from '@/store/windows'

import { PreviewPane } from './preview-pane'

interface PreviewTilePaneProps {
  /** The tab's own Close, for body states that offer one (a failed load). */
  onClose?: () => void
  /** The `$previewTabs` id this pane renders. */
  tabId: string
}

/**
 * One preview, as a layout-tree pane. The tab strip — its label, close verbs,
 * drag/stack/split and ⌘W — belongs to the ZONE (see `preview-tile.tsx`), so
 * this renders only the body and a preview tab behaves like every other tab.
 *
 * The console / DevTools toggles live in the pane's own browser bar beside the
 * address (`preview-browser-bar`); the restart handler arrives through the atom
 * bridge the old rail wrapper used, since the mirror renders this pane with no
 * props to thread.
 */
export function PreviewTilePane({ onClose, tabId }: PreviewTilePaneProps) {
  const previewReloadRequest = useStore($previewReloadRequest)
  const previewTabs = useStore($previewTabs)
  const restartPreviewServer = useStore($restartPreviewServer)

  // A popped-out Browser is a fresh renderer: no session ever pushes a scope
  // there, so its scoped view may start empty. Pull this window's tab in from
  // shared storage; the docked mirror never runs this (it is not a browser
  // window), so a closed tab stays closed there.
  useEffect(() => {
    if (isBrowserWindow() && tabId && !previewTabs.some(tab => tab.id === tabId)) {
      adoptPersistedBrowserTab(tabId)
    }
  }, [previewTabs, tabId])

  // A hidden page of another profile's chat (the rail now shows the next bot's
  // bucket) is still an open tab: read it from its own bucket, or this parked
  // body would render nothing and tear its guest down.
  const tab = previewTabs.find(candidate => candidate.id === tabId) ?? previewTabInAnyProfile(tabId)
  const target = tab?.target

  // The tab closed while this pane was still mounted (the mirror disposes it a
  // tick later).
  if (!target) {
    return null
  }

  return (
    <PreviewPane
      embedded
      onClose={onClose}
      onRestartServer={target.kind === 'url' ? (restartPreviewServer ?? undefined) : undefined}
      reloadRequest={previewReloadRequest}
      sessionId={tab?.sessionId}
      tabId={tabId}
      target={target}
    />
  )
}

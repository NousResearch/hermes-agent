import { useStore } from '@nanostores/react'

import { $restartPreviewServer } from '@/app/contrib/panes'
import { $previewReloadRequest, $previewTabs } from '@/store/preview'

import { PreviewPane } from './preview-pane'

interface PreviewTilePaneProps {
  /** The `$previewTabs` id this pane renders. */
  tabId: string
}

/**
 * One preview, as a layout-tree pane. The tab strip — its label, close verbs,
 * drag/stack/split and ⌘W — belongs to the ZONE (see `preview-tile.tsx`), so
 * this renders only the body and a preview tab behaves like every other tab.
 *
 * The console / DevTools glyphs live in the zone STRIP (`preview-strip-tools`),
 * keyed by `tabId`; the restart handler arrives through the atom bridge the old
 * rail wrapper used, since the mirror renders this pane with no props to thread.
 */
export function PreviewTilePane({ tabId }: PreviewTilePaneProps) {
  const previewReloadRequest = useStore($previewReloadRequest)
  const previewTabs = useStore($previewTabs)
  const restartPreviewServer = useStore($restartPreviewServer)
  const target = previewTabs.find(tab => tab.id === tabId)?.target

  // The tab closed while this pane was still mounted (the mirror disposes it a
  // tick later).
  if (!target) {
    return null
  }

  return (
    <PreviewPane
      embedded
      onRestartServer={target.kind === 'url' ? (restartPreviewServer ?? undefined) : undefined}
      reloadRequest={previewReloadRequest}
      tabId={tabId}
      target={target}
    />
  )
}

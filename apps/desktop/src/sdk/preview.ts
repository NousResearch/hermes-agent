import type { PluginSessionContext } from '@/contrib/session'
import { resolveSessionContributionContext } from '@/contrib/session-context'
import { $browserPages, $previewTabs, type BrowserDocument, openPreview } from '@/store/preview'
import { ownerLookupSessionRows, sessionMatchesStoredId } from '@/store/session'

import { safeViewerUrl, sameViewerLocation } from '../../electron/plugin-viewer-policy'

import { startViewerKeepAlive } from './viewer-keep-alive'

export interface PluginPreviewInput {
  /** Client-reachable absolute HTTP(S) URL. No ambient auth headers are added. */
  url: string
  label?: string
  session: PluginSessionContext
  /** Renew a scoped lease while this original tab/document remains open. Runs in the host renderer. */
  onKeepAlive?: () => Promise<void>
}

export function currentPluginSession(session: PluginSessionContext): boolean {
  if (!session || !session.connectionId || !session.profile) {
    return false
  }

  const row = ownerLookupSessionRows().find(
    row =>
      sessionMatchesStoredId(row, session.storedSessionId || '') &&
      row.connection_id === session.connectionId &&
      row.profile === session.profile
  )

  const current = resolveSessionContributionContext({ storedSessionId: session.storedSessionId, row })

  return Boolean(
    current &&
    current.connectionId === session.connectionId &&
    current.profile === session.profile &&
    current.storedSessionId === session.storedSessionId &&
    current.runtimeSessionId === session.runtimeSessionId
  )
}

/** Explicit UI action, not an event handler. Ticket-bearing tabs never persist. */
export async function openPluginPreview(input: PluginPreviewInput): Promise<boolean> {
  const url = safeViewerUrl(input?.url)

  if (!url || !currentPluginSession(input.session)) {
    return false
  }

  const tab = openPreview(
    { kind: 'url', url, source: url, label: input.label || 'Viewer', transient: true, browserContext: 'isolated' },
    'explicit-link'
  )

  if (input.onKeepAlive) {
    const onKeepAlive = input.onKeepAlive
    let document: BrowserDocument | undefined
    let stopped = false
    let stopRenewal: (() => void) | undefined
    const subscriptions: Array<() => void> = []

    const stop = () => {
      if (stopped) {
        return
      }

      stopped = true
      stopRenewal?.()
      subscriptions.forEach(unsubscribe => unsubscribe())
    }

    const isOpen = () => {
      const page = $browserPages.get()[tab.id]

      return (
        !stopped &&
        $previewTabs.get().includes(tab) &&
        tab.target.transient === true &&
        document !== undefined &&
        page?.document === document &&
        document.isLive() &&
        sameViewerLocation(url, page.url)
      )
    }

    const check = () => {
      if (stopped) {
        return
      }

      if (!$previewTabs.get().includes(tab)) {
        stop()

        return
      }

      if (!document) {
        const page = $browserPages.get()[tab.id]

        if (!page?.document?.isLive()) {
          return
        }

        document = page.document

        if (isOpen()) {
          stopRenewal = startViewerKeepAlive({ isOpen, onKeepAlive, onStop: stop })

          return
        }
      }

      if (document && !isOpen()) {
        stop()
      }
    }

    subscriptions.push($previewTabs.listen(check), $browserPages.listen(check))
    check()
  }

  return true
}
